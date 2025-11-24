"""
Manual ROI Annotation Tool for Voltage Imaging
Interactive tool for drawing polygon ROIs around neurons
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import tifffile
from pathlib import Path
from datetime import datetime

from annotation_schema import (
    VideoAnnotation, ROIAnnotation, ROIQuality, ROIShape
)


class ROIAnnotator:
    """Interactive ROI annotation tool with polygon drawing"""

    def __init__(self, video_path: str, fps: float = 200):
        """
        Initialize the ROI annotator

        Parameters:
        -----------
        video_path : str
            Path to the TIFF video file
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

        # Compute projections
        print("Computing projections...")
        self.mean_proj = np.mean(self.video, axis=0)
        self.std_proj = np.std(self.video, axis=0)
        self.max_proj = np.max(self.video, axis=0)

        # Current projection being displayed
        self.current_proj = 'std'
        self.projections = {
            'mean': self.mean_proj,
            'std': self.std_proj,
            'max': self.max_proj
        }

        # Initialize or load annotations
        self.annotation = VideoAnnotation(
            video_path=str(video_path),
            video_name=self.video_name,
            n_frames=self.n_frames,
            height=self.height,
            width=self.width,
            fps=fps
        )

        # Drawing state
        self.current_polygon = []  # Points being drawn
        self.drawing = False
        self.roi_patches = {}  # roi_id -> patch
        self.roi_texts = {}  # roi_id -> text annotation

        # Current ROI properties
        self.current_quality = ROIQuality.GOOD.value
        self.current_shape = ROIShape.RING.value

        # Undo stack
        self.undo_stack = []

        # Selected ROI for editing
        self.selected_roi_id = None

    def _compute_centroid(self, polygon):
        """Compute centroid of polygon"""
        polygon = np.array(polygon)
        return [float(np.mean(polygon[:, 0])), float(np.mean(polygon[:, 1]))]

    def _compute_bbox(self, polygon):
        """Compute bounding box [x_min, y_min, x_max, y_max]"""
        polygon = np.array(polygon)
        return [
            float(np.min(polygon[:, 0])),
            float(np.min(polygon[:, 1])),
            float(np.max(polygon[:, 0])),
            float(np.max(polygon[:, 1]))
        ]

    def _compute_area(self, polygon):
        """Compute polygon area using shoelace formula"""
        polygon = np.array(polygon)
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i, 0] * polygon[j, 1]
            area -= polygon[j, 0] * polygon[i, 1]
        return abs(area) / 2.0

    def _get_next_roi_id(self):
        """Get next available ROI ID"""
        if not self.annotation.rois:
            return 0
        return max(r.roi_id for r in self.annotation.rois) + 1

    def _draw_roi(self, roi: ROIAnnotation, color='cyan', alpha=0.3):
        """Draw a single ROI on the axes"""
        polygon = np.array(roi.polygon)

        # Create patch
        patch = MplPolygon(
            polygon,
            fill=True,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=2
        )
        self.ax.add_patch(patch)
        self.roi_patches[roi.roi_id] = patch

        # Add label
        centroid = roi.centroid
        text = self.ax.text(
            centroid[0], centroid[1] - 10,
            f"{roi.roi_id}",
            color='white',
            fontsize=10,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
        )
        self.roi_texts[roi.roi_id] = text

    def _redraw_all_rois(self):
        """Redraw all ROIs"""
        # Clear existing patches and texts
        for patch in self.roi_patches.values():
            patch.remove()
        for text in self.roi_texts.values():
            text.remove()
        self.roi_patches.clear()
        self.roi_texts.clear()

        # Draw all ROIs
        for roi in self.annotation.rois:
            # Color by quality
            if roi.quality == ROIQuality.GOOD.value:
                color = 'cyan'
            elif roi.quality == ROIQuality.UNCERTAIN.value:
                color = 'yellow'
            elif roi.quality == ROIQuality.OVERLAPPING.value:
                color = 'orange'
            else:
                color = 'red'

            # Highlight selected ROI
            if roi.roi_id == self.selected_roi_id:
                self._draw_roi(roi, color='lime', alpha=0.5)
            else:
                self._draw_roi(roi, color=color, alpha=0.3)

        self.fig.canvas.draw_idle()

    def _update_projection(self, label):
        """Switch projection view"""
        self.current_proj = label
        self.im.set_data(self.projections[label])
        self.im.set_clim(vmin=np.percentile(self.projections[label], 1),
                        vmax=np.percentile(self.projections[label], 99))
        self.ax.set_title(f'{label.upper()} Projection - {len(self.annotation.rois)} ROIs')
        self.fig.canvas.draw_idle()

    def _update_quality(self, label):
        """Update current quality setting"""
        self.current_quality = label

    def _update_shape(self, label):
        """Update current shape setting"""
        self.current_shape = label

    def _on_click(self, event):
        """Handle mouse click for polygon drawing"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click - add point
            self.current_polygon.append([event.xdata, event.ydata])

            # Draw point
            self.ax.plot(event.xdata, event.ydata, 'r.', markersize=8)

            # Draw line to previous point
            if len(self.current_polygon) > 1:
                pts = np.array(self.current_polygon[-2:])
                self.ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=2)

            self.fig.canvas.draw_idle()

        elif event.button == 3:  # Right click - complete polygon
            if len(self.current_polygon) >= 3:
                self._complete_polygon()

    def _on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'enter':
            # Complete current polygon
            if len(self.current_polygon) >= 3:
                self._complete_polygon()

        elif event.key == 'escape':
            # Cancel current polygon
            self.current_polygon = []
            self._redraw_all_rois()
            self._update_status("Polygon cancelled")

        elif event.key == 'z' and event.key == 'ctrl+z':
            # Undo
            self._undo()

        elif event.key == 'delete' or event.key == 'd':
            # Delete selected ROI
            if self.selected_roi_id is not None:
                self._delete_roi(self.selected_roi_id)

        elif event.key == 's':
            # Save
            self._save_annotations()

        elif event.key == '1':
            self._update_projection('mean')
        elif event.key == '2':
            self._update_projection('std')
        elif event.key == '3':
            self._update_projection('max')

    def _complete_polygon(self):
        """Complete the current polygon and create ROI"""
        if len(self.current_polygon) < 3:
            return

        # Create ROI annotation
        roi_id = self._get_next_roi_id()
        roi = ROIAnnotation(
            roi_id=roi_id,
            polygon=self.current_polygon.copy(),
            centroid=self._compute_centroid(self.current_polygon),
            bbox=self._compute_bbox(self.current_polygon),
            quality=self.current_quality,
            shape=self.current_shape,
            area_pixels=self._compute_area(self.current_polygon)
        )

        # Add to annotation
        self.annotation.add_roi(roi)

        # Add to undo stack
        self.undo_stack.append(('add', roi_id))

        # Reset drawing state
        self.current_polygon = []

        # Redraw
        self._redraw_all_rois()
        self._update_status(f"ROI {roi_id} created ({roi.area_pixels:.0f} px)")

    def _delete_roi(self, roi_id: int):
        """Delete an ROI"""
        roi = self.annotation.get_roi_by_id(roi_id)
        if roi:
            # Add to undo stack
            self.undo_stack.append(('delete', roi.to_dict()))

            # Remove
            self.annotation.remove_roi(roi_id)
            self.selected_roi_id = None

            # Redraw
            self._redraw_all_rois()
            self._update_status(f"ROI {roi_id} deleted")

    def _undo(self):
        """Undo last action"""
        if not self.undo_stack:
            self._update_status("Nothing to undo")
            return

        action, data = self.undo_stack.pop()

        if action == 'add':
            # Undo add by deleting
            self.annotation.remove_roi(data)
            self._update_status(f"Undone: ROI {data} removed")

        elif action == 'delete':
            # Undo delete by re-adding
            roi = ROIAnnotation.from_dict(data)
            self.annotation.rois.append(roi)
            self._update_status(f"Undone: ROI {data['roi_id']} restored")

        self._redraw_all_rois()

    def _select_roi_at_point(self, x, y):
        """Select ROI that contains the given point"""
        from matplotlib.path import Path

        for roi in self.annotation.rois:
            polygon_path = Path(roi.polygon)
            if polygon_path.contains_point([x, y]):
                self.selected_roi_id = roi.roi_id
                self._redraw_all_rois()
                self._update_status(f"Selected ROI {roi.roi_id} (quality: {roi.quality})")
                return

        # No ROI at point
        self.selected_roi_id = None
        self._redraw_all_rois()

    def _on_double_click(self, event):
        """Handle double click to select ROI"""
        if event.inaxes != self.ax:
            return
        if event.dblclick:
            self._select_roi_at_point(event.xdata, event.ydata)

    def _save_annotations(self, event=None):
        """Save annotations to file"""
        output_path = f"annotations/{self.video_name}_rois.json"
        self.annotation.save(output_path)
        self._update_status(f"Saved to {output_path}")

    def _load_annotations(self, path: str):
        """Load existing annotations"""
        try:
            self.annotation = VideoAnnotation.load(path)
            self._redraw_all_rois()
            self._update_status(f"Loaded {len(self.annotation.rois)} ROIs from {path}")
        except FileNotFoundError:
            self._update_status(f"No existing annotations found")

    def _update_status(self, message: str):
        """Update status text"""
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()

    def _show_help(self, event=None):
        """Show help dialog"""
        help_text = """
ROI Annotator Controls:
─────────────────────────────
LEFT CLICK: Add point to polygon
RIGHT CLICK / ENTER: Complete polygon
DOUBLE CLICK: Select existing ROI
DELETE / D: Delete selected ROI
ESCAPE: Cancel current polygon
S: Save annotations

1/2/3: Switch projection (mean/std/max)
CTRL+Z: Undo last action

Tips:
- Use STD projection to see active neurons
- Draw polygons clockwise
- Mark quality for each ROI
─────────────────────────────
        """
        print(help_text)
        self._update_status("Help printed to console")

    def run(self):
        """Launch the interactive annotator"""
        # Create figure
        self.fig = plt.figure(figsize=(16, 10))

        # Main image axes
        self.ax = self.fig.add_axes([0.05, 0.15, 0.65, 0.8])

        # Display std projection by default
        self.im = self.ax.imshow(
            self.std_proj,
            cmap='hot',
            vmin=np.percentile(self.std_proj, 1),
            vmax=np.percentile(self.std_proj, 99)
        )
        self.ax.set_title(f'STD Projection - {len(self.annotation.rois)} ROIs')
        self.ax.set_xlabel('X (pixels)')
        self.ax.set_ylabel('Y (pixels)')

        # Status text
        self.status_text = self.fig.text(
            0.05, 0.02,
            "Left click to add points, right click to complete polygon",
            fontsize=10
        )

        # Projection selector
        ax_proj = self.fig.add_axes([0.75, 0.7, 0.2, 0.15])
        self.radio_proj = RadioButtons(ax_proj, ('mean', 'std', 'max'), active=1)
        self.radio_proj.on_clicked(self._update_projection)
        ax_proj.set_title('Projection')

        # Quality selector
        ax_quality = self.fig.add_axes([0.75, 0.5, 0.2, 0.15])
        qualities = [q.value for q in ROIQuality]
        self.radio_quality = RadioButtons(ax_quality, qualities, active=0)
        self.radio_quality.on_clicked(self._update_quality)
        ax_quality.set_title('ROI Quality')

        # Shape selector
        ax_shape = self.fig.add_axes([0.75, 0.3, 0.2, 0.15])
        shapes = [s.value for s in ROIShape]
        self.radio_shape = RadioButtons(ax_shape, shapes, active=0)
        self.radio_shape.on_clicked(self._update_shape)
        ax_shape.set_title('ROI Shape')

        # Buttons
        ax_save = self.fig.add_axes([0.75, 0.15, 0.1, 0.05])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self._save_annotations)

        ax_help = self.fig.add_axes([0.86, 0.15, 0.1, 0.05])
        self.btn_help = Button(ax_help, 'Help')
        self.btn_help.on_clicked(self._show_help)

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_press_event', self._on_double_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Try to load existing annotations
        existing_path = f"annotations/{self.video_name}_rois.json"
        if Path(existing_path).exists():
            self._load_annotations(existing_path)

        # Draw existing ROIs
        self._redraw_all_rois()

        print("\n" + "="*50)
        print("ROI ANNOTATOR")
        print("="*50)
        print("Controls:")
        print("  LEFT CLICK: Add point to polygon")
        print("  RIGHT CLICK / ENTER: Complete polygon")
        print("  DOUBLE CLICK: Select ROI")
        print("  DELETE: Remove selected ROI")
        print("  S: Save | ESC: Cancel | 1/2/3: Switch view")
        print("="*50 + "\n")

        plt.show()

        return self.annotation


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default path
        video_path = "Voltron2-ST_fish2_fov25_spin_conf-5ms_fTL_5_13_25_027_crop_cleaned.tif"

    fps = 200
    if len(sys.argv) > 2:
        fps = float(sys.argv[2])

    annotator = ROIAnnotator(video_path, fps=fps)
    annotation = annotator.run()

    print("\n" + annotation.summary())


if __name__ == "__main__":
    main()
