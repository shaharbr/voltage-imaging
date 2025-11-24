"""
Manual Spike Annotation Tool for Voltage Imaging
Interactive tool for marking spike times on voltage traces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons, SpanSelector
from matplotlib.patches import Rectangle
import tifffile
from pathlib import Path
from scipy import signal

from annotation_schema import (
    VideoAnnotation, SpikeAnnotation, SpikeConfidence
)


class SpikeAnnotator:
    """Interactive spike annotation tool"""

    def __init__(self, video_path: str, roi_annotation_path: str = None, fps: float = 200):
        """
        Initialize the spike annotator

        Parameters:
        -----------
        video_path : str
            Path to the TIFF video file
        roi_annotation_path : str
            Path to ROI annotations JSON (if None, will look for default location)
        fps : float
            Frame rate in Hz
        """
        self.video_path = video_path
        self.fps = fps
        self.video_name = Path(video_path).stem

        # Load video
        print(f"Loading video: {video_path}")
        self.video = tifffile.imread(video_path)
        self.n_frames, self.height, self.width = self.video.shape
        print(f"Loaded: {self.n_frames} frames, {self.height}x{self.width}")

        # Load ROI annotations
        if roi_annotation_path is None:
            roi_annotation_path = f"annotations/{self.video_name}_rois.json"

        print(f"Loading ROI annotations from: {roi_annotation_path}")
        self.annotation = VideoAnnotation.load(roi_annotation_path)
        print(f"Loaded {len(self.annotation.rois)} ROIs")

        # Extract traces for all ROIs
        self._extract_traces()

        # Current state
        self.current_roi_idx = 0
        self.time_axis = np.arange(self.n_frames) / self.fps

        # View window (in seconds)
        self.view_start = 0
        self.view_duration = 5.0  # Show 5 seconds at a time

        # Auto-detection suggestions
        self.auto_spikes = {}
        self.show_auto_spikes = True

        # Current confidence level
        self.current_confidence = SpikeConfidence.CERTAIN.value

    def _extract_traces(self):
        """Extract fluorescence traces for all ROIs"""
        print("Extracting traces...")
        from matplotlib.path import Path as MplPath

        self.raw_traces = {}
        self.processed_traces = {}

        for roi in self.annotation.rois:
            # Create mask from polygon
            polygon = np.array(roi.polygon)
            mask = np.zeros((self.height, self.width), dtype=bool)

            # Fill polygon mask
            y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
            points = np.vstack((x_coords.ravel(), y_coords.ravel())).T
            path = MplPath(polygon)
            mask = path.contains_points(points).reshape(self.height, self.width)

            # Extract trace
            trace = np.zeros(self.n_frames)
            for t in range(self.n_frames):
                frame = self.video[t]
                trace[t] = np.mean(frame[mask])

            self.raw_traces[roi.roi_id] = trace

            # Process trace (dF/F and detrend)
            f0 = np.percentile(trace, 10)
            dff = (trace - f0) / f0
            dff_detrend = signal.detrend(dff)
            self.processed_traces[roi.roi_id] = dff_detrend

        print(f"Extracted {len(self.raw_traces)} traces")

    def _auto_detect_spikes(self, roi_id: int, threshold_std: float = 3.0):
        """Auto-detect spikes for suggestions (not final annotations)"""
        trace = self.processed_traces[roi_id]

        # Invert for negative deflections (Voltron-2)
        inverted = -trace

        # Find peaks
        noise = np.std(inverted[inverted < np.percentile(inverted, 50)])
        threshold = np.median(inverted) + threshold_std * noise

        from scipy.signal import find_peaks
        peaks, properties = find_peaks(
            inverted,
            height=threshold,
            distance=int(0.005 * self.fps),  # 5ms minimum
            prominence=0.5 * noise
        )

        self.auto_spikes[roi_id] = peaks / self.fps  # Convert to seconds
        return self.auto_spikes[roi_id]

    def _get_spikes_for_roi(self, roi_id: int):
        """Get manually annotated spikes for ROI"""
        return [s for s in self.annotation.spikes if s.roi_id == roi_id]

    def _get_next_spike_id(self):
        """Get next available spike ID"""
        if not self.annotation.spikes:
            return 0
        return max(s.spike_id for s in self.annotation.spikes) + 1

    def _add_spike(self, time_seconds: float, roi_id: int):
        """Add a spike annotation"""
        # Check if spike already exists nearby (within 5ms)
        existing = self._get_spikes_for_roi(roi_id)
        for s in existing:
            if abs(s.time_seconds - time_seconds) < 0.005:
                self._update_status(f"Spike already exists at {s.time_seconds:.3f}s")
                return None

        # Get amplitude from trace
        frame_idx = int(time_seconds * self.fps)
        frame_idx = max(0, min(frame_idx, self.n_frames - 1))
        amplitude = self.processed_traces[roi_id][frame_idx]

        spike = SpikeAnnotation(
            spike_id=self._get_next_spike_id(),
            roi_id=roi_id,
            time_seconds=time_seconds,
            frame_index=frame_idx,
            amplitude=amplitude,
            confidence=self.current_confidence,
            source="manual"
        )

        self.annotation.add_spike(spike)
        self._update_status(f"Added spike at {time_seconds:.3f}s (amp: {amplitude:.3f})")
        return spike

    def _remove_spike_at(self, time_seconds: float, roi_id: int, tolerance: float = 0.01):
        """Remove spike near given time"""
        for spike in self._get_spikes_for_roi(roi_id):
            if abs(spike.time_seconds - time_seconds) < tolerance:
                self.annotation.remove_spike(spike.spike_id)
                self._update_status(f"Removed spike at {spike.time_seconds:.3f}s")
                return True
        return False

    def _accept_auto_spike(self, time_seconds: float, roi_id: int):
        """Accept an auto-detected spike"""
        spike = self._add_spike(time_seconds, roi_id)
        if spike:
            spike.source = "auto_accepted"

    def _update_plot(self):
        """Update the trace plot"""
        roi = self.annotation.rois[self.current_roi_idx]
        roi_id = roi.roi_id
        trace = self.processed_traces[roi_id]

        # Clear axes
        self.ax_trace.clear()
        self.ax_overview.clear()

        # Time axis
        time = self.time_axis

        # --- Main trace view (zoomed) ---
        view_mask = (time >= self.view_start) & (time < self.view_start + self.view_duration)
        view_time = time[view_mask]
        view_trace = trace[view_mask]

        self.ax_trace.plot(view_time, view_trace, 'k-', linewidth=0.8)
        self.ax_trace.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Plot manual spikes
        manual_spikes = self._get_spikes_for_roi(roi_id)
        for spike in manual_spikes:
            if self.view_start <= spike.time_seconds < self.view_start + self.view_duration:
                # Color by confidence
                if spike.confidence == SpikeConfidence.CERTAIN.value:
                    color = 'red'
                elif spike.confidence == SpikeConfidence.PROBABLE.value:
                    color = 'orange'
                else:
                    color = 'yellow'

                self.ax_trace.axvline(spike.time_seconds, color=color, alpha=0.7, linewidth=2)
                self.ax_trace.plot(spike.time_seconds, spike.amplitude, 'o',
                                  color=color, markersize=10, markeredgecolor='black')

        # Plot auto-detected spikes (suggestions)
        if self.show_auto_spikes and roi_id in self.auto_spikes:
            for t in self.auto_spikes[roi_id]:
                if self.view_start <= t < self.view_start + self.view_duration:
                    # Check if already manually annotated
                    already_annotated = any(abs(s.time_seconds - t) < 0.005 for s in manual_spikes)
                    if not already_annotated:
                        idx = int(t * self.fps)
                        self.ax_trace.axvline(t, color='cyan', alpha=0.3, linewidth=1)
                        self.ax_trace.plot(t, trace[idx], '^', color='cyan',
                                          markersize=6, alpha=0.5)

        self.ax_trace.set_xlabel('Time (s)')
        self.ax_trace.set_ylabel('ΔF/F')
        self.ax_trace.set_title(
            f'ROI {roi_id} (quality: {roi.quality}) - '
            f'{len(manual_spikes)} spikes annotated'
        )
        self.ax_trace.set_xlim(self.view_start, self.view_start + self.view_duration)

        # --- Overview (full trace) ---
        self.ax_overview.plot(time, trace, 'k-', linewidth=0.3)

        # Show current view window
        self.ax_overview.axvspan(self.view_start, self.view_start + self.view_duration,
                                 color='blue', alpha=0.2)

        # Mark all spikes on overview
        for spike in manual_spikes:
            self.ax_overview.axvline(spike.time_seconds, color='red', alpha=0.5, linewidth=0.5)

        self.ax_overview.set_xlabel('Time (s)')
        self.ax_overview.set_ylabel('ΔF/F')
        self.ax_overview.set_title('Full trace overview (blue = current view)')

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handle click on trace to add/remove spike"""
        if event.inaxes != self.ax_trace:
            return

        roi_id = self.annotation.rois[self.current_roi_idx].roi_id

        if event.button == 1:  # Left click - add spike
            self._add_spike(event.xdata, roi_id)
            self._update_plot()

        elif event.button == 3:  # Right click - remove spike
            self._remove_spike_at(event.xdata, roi_id)
            self._update_plot()

    def _on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'right':
            # Next time window
            self.view_start = min(self.view_start + self.view_duration * 0.5,
                                  self.time_axis[-1] - self.view_duration)
            self._update_plot()

        elif event.key == 'left':
            # Previous time window
            self.view_start = max(self.view_start - self.view_duration * 0.5, 0)
            self._update_plot()

        elif event.key == 'up':
            # Previous ROI
            self.current_roi_idx = (self.current_roi_idx - 1) % len(self.annotation.rois)
            self._run_auto_detect()
            self._update_plot()

        elif event.key == 'down':
            # Next ROI
            self.current_roi_idx = (self.current_roi_idx + 1) % len(self.annotation.rois)
            self._run_auto_detect()
            self._update_plot()

        elif event.key == 's':
            # Save
            self._save_annotations()

        elif event.key == 'a':
            # Accept all auto-detected spikes in current view
            self._accept_visible_auto_spikes()

        elif event.key == '+' or event.key == '=':
            # Zoom in (shorter window)
            self.view_duration = max(0.5, self.view_duration / 2)
            self._update_plot()

        elif event.key == '-':
            # Zoom out (longer window)
            self.view_duration = min(30, self.view_duration * 2)
            self._update_plot()

        elif event.key == '1':
            self.current_confidence = SpikeConfidence.CERTAIN.value
            self._update_status("Confidence: CERTAIN")
        elif event.key == '2':
            self.current_confidence = SpikeConfidence.PROBABLE.value
            self._update_status("Confidence: PROBABLE")
        elif event.key == '3':
            self.current_confidence = SpikeConfidence.UNCERTAIN.value
            self._update_status("Confidence: UNCERTAIN")

    def _on_scroll(self, event):
        """Handle scroll to navigate time"""
        if event.inaxes in [self.ax_trace, self.ax_overview]:
            if event.button == 'up':
                self.view_start = max(self.view_start - self.view_duration * 0.2, 0)
            else:
                self.view_start = min(self.view_start + self.view_duration * 0.2,
                                      self.time_axis[-1] - self.view_duration)
            self._update_plot()

    def _run_auto_detect(self):
        """Run auto-detection for current ROI"""
        roi_id = self.annotation.rois[self.current_roi_idx].roi_id
        if roi_id not in self.auto_spikes:
            self._auto_detect_spikes(roi_id)

    def _accept_visible_auto_spikes(self):
        """Accept all auto-detected spikes in current view window"""
        roi_id = self.annotation.rois[self.current_roi_idx].roi_id
        if roi_id not in self.auto_spikes:
            return

        count = 0
        for t in self.auto_spikes[roi_id]:
            if self.view_start <= t < self.view_start + self.view_duration:
                # Check if already annotated
                existing = self._get_spikes_for_roi(roi_id)
                if not any(abs(s.time_seconds - t) < 0.005 for s in existing):
                    self._accept_auto_spike(t, roi_id)
                    count += 1

        self._update_status(f"Accepted {count} auto-detected spikes")
        self._update_plot()

    def _save_annotations(self, event=None):
        """Save annotations"""
        output_path = f"annotations/{self.video_name}_annotations.json"
        self.annotation.save(output_path)
        self._update_status(f"Saved to {output_path}")

    def _toggle_auto_spikes(self, label):
        """Toggle auto-spike suggestions"""
        self.show_auto_spikes = not self.show_auto_spikes
        self._update_plot()

    def _update_status(self, message: str):
        """Update status text"""
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()
        print(message)

    def _show_help(self, event=None):
        """Show help"""
        help_text = """
Spike Annotator Controls:
─────────────────────────────
LEFT CLICK: Add spike at cursor
RIGHT CLICK: Remove nearest spike

UP/DOWN: Switch ROI
LEFT/RIGHT: Navigate time
SCROLL: Pan through trace
+/-: Zoom in/out

A: Accept all auto-spikes in view
S: Save annotations

1/2/3: Set confidence (certain/probable/uncertain)

Colors:
- RED: Certain spike
- ORANGE: Probable spike
- YELLOW: Uncertain spike
- CYAN: Auto-detected suggestion
─────────────────────────────
        """
        print(help_text)

    def run(self):
        """Launch the interactive annotator"""
        # Create figure
        self.fig = plt.figure(figsize=(16, 10))

        # Trace axes (main view)
        self.ax_trace = self.fig.add_axes([0.08, 0.35, 0.85, 0.55])

        # Overview axes
        self.ax_overview = self.fig.add_axes([0.08, 0.08, 0.85, 0.2])

        # Status text
        self.status_text = self.fig.text(
            0.08, 0.96,
            "Left click to add spike, right click to remove",
            fontsize=10, fontweight='bold'
        )

        # Buttons
        ax_save = self.fig.add_axes([0.85, 0.92, 0.08, 0.04])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self._save_annotations)

        ax_help = self.fig.add_axes([0.75, 0.92, 0.08, 0.04])
        self.btn_help = Button(ax_help, 'Help')
        self.btn_help.on_clicked(self._show_help)

        # Auto-spike toggle
        ax_auto = self.fig.add_axes([0.60, 0.92, 0.12, 0.04])
        self.check_auto = CheckButtons(ax_auto, ['Show auto'], [True])
        self.check_auto.on_clicked(self._toggle_auto_spikes)

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

        # Run auto-detection for first ROI
        self._run_auto_detect()

        # Initial plot
        self._update_plot()

        print("\n" + "="*50)
        print("SPIKE ANNOTATOR")
        print("="*50)
        print("Controls:")
        print("  LEFT CLICK: Add spike")
        print("  RIGHT CLICK: Remove spike")
        print("  UP/DOWN: Switch ROI")
        print("  LEFT/RIGHT: Navigate | +/-: Zoom")
        print("  A: Accept auto-spikes | S: Save")
        print("="*50 + "\n")

        plt.show()

        return self.annotation


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "Voltron2-ST_fish2_fov25_spin_conf-5ms_fTL_5_13_25_027_crop_cleaned.tif"

    roi_path = None
    if len(sys.argv) > 2:
        roi_path = sys.argv[2]

    fps = 200
    if len(sys.argv) > 3:
        fps = float(sys.argv[3])

    annotator = SpikeAnnotator(video_path, roi_path, fps=fps)
    annotation = annotator.run()

    print("\n" + annotation.summary())


if __name__ == "__main__":
    main()
