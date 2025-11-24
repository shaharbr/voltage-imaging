"""
Annotation Visualization and Review Tool
Generates summary figures and allows review of annotations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from matplotlib.gridspec import GridSpec
import tifffile
from pathlib import Path
from scipy import signal
import pandas as pd

from annotation_schema import VideoAnnotation, ROIQuality, SpikeConfidence


class AnnotationViewer:
    """Visualization and review tool for annotations"""

    def __init__(self, annotation_path: str, video_path: str = None):
        """
        Initialize viewer

        Parameters:
        -----------
        annotation_path : str
            Path to annotation JSON file
        video_path : str
            Path to video file (optional, will use path from annotation if not provided)
        """
        print(f"Loading annotations from: {annotation_path}")
        self.annotation = VideoAnnotation.load(annotation_path)

        if video_path is None:
            video_path = self.annotation.video_path

        print(f"Loading video: {video_path}")
        self.video = tifffile.imread(video_path)
        self.n_frames, self.height, self.width = self.video.shape

        # Compute projections
        self.mean_proj = np.mean(self.video, axis=0)
        self.std_proj = np.std(self.video, axis=0)

        # Extract traces
        self._extract_traces()

    def _extract_traces(self):
        """Extract traces for all ROIs"""
        self.traces = {}

        for roi in self.annotation.rois:
            polygon = np.array(roi.polygon)
            mask = np.zeros((self.height, self.width), dtype=bool)

            y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
            points = np.vstack((x_coords.ravel(), y_coords.ravel())).T
            path = MplPath(polygon)
            mask = path.contains_points(points).reshape(self.height, self.width)

            trace = np.zeros(self.n_frames)
            for t in range(self.n_frames):
                trace[t] = np.mean(self.video[t][mask])

            # Process
            f0 = np.percentile(trace, 10)
            dff = (trace - f0) / f0
            self.traces[roi.roi_id] = signal.detrend(dff)

    def plot_roi_overview(self, save_path: str = None):
        """Generate overview figure showing all ROIs"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Mean projection with ROIs
        axes[0].imshow(self.mean_proj, cmap='gray',
                      vmin=np.percentile(self.mean_proj, 1),
                      vmax=np.percentile(self.mean_proj, 99))
        axes[0].set_title('Mean Projection')

        # Std projection with ROIs
        axes[1].imshow(self.std_proj, cmap='hot',
                      vmin=np.percentile(self.std_proj, 1),
                      vmax=np.percentile(self.std_proj, 99))
        axes[1].set_title('Std Projection')

        # Draw ROIs on both
        quality_colors = {
            ROIQuality.GOOD.value: 'cyan',
            ROIQuality.UNCERTAIN.value: 'yellow',
            ROIQuality.OVERLAPPING.value: 'orange',
            ROIQuality.PARTIAL.value: 'red'
        }

        for ax in axes[:2]:
            for roi in self.annotation.rois:
                color = quality_colors.get(roi.quality, 'cyan')
                polygon = MplPolygon(
                    roi.polygon,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.5
                )
                ax.add_patch(polygon)
                ax.text(roi.centroid[0], roi.centroid[1] - 8,
                       str(roi.roi_id), color='white', fontsize=8,
                       ha='center', fontweight='bold')

        # Quality distribution
        quality_counts = {}
        for roi in self.annotation.rois:
            quality_counts[roi.quality] = quality_counts.get(roi.quality, 0) + 1

        labels = list(quality_counts.keys())
        sizes = list(quality_counts.values())
        colors = [quality_colors.get(q, 'gray') for q in labels]

        axes[2].pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%')
        axes[2].set_title(f'ROI Quality Distribution (n={len(self.annotation.rois)})')

        plt.suptitle(f'{self.annotation.video_name} - ROI Overview', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()
        return fig

    def plot_all_traces(self, n_cols: int = 4, save_path: str = None):
        """Plot all traces in a grid"""
        n_rois = len(self.annotation.rois)
        n_rows = int(np.ceil(n_rois / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2 * n_rows),
                                sharex=True, sharey=True)
        axes = axes.flatten() if n_rois > 1 else [axes]

        time = np.arange(self.n_frames) / self.annotation.fps

        for idx, roi in enumerate(self.annotation.rois):
            ax = axes[idx]
            trace = self.traces[roi.roi_id]

            ax.plot(time, trace, 'k-', linewidth=0.5)

            # Plot spikes
            spikes = self.annotation.get_spikes_for_roi(roi.roi_id)
            for spike in spikes:
                color = 'red' if spike.confidence == SpikeConfidence.CERTAIN.value else 'orange'
                ax.axvline(spike.time_seconds, color=color, alpha=0.5, linewidth=1)

            ax.set_title(f'ROI {roi.roi_id} ({len(spikes)} spikes)', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Hide empty axes
        for idx in range(n_rois, len(axes)):
            axes[idx].set_visible(False)

        # Labels
        fig.text(0.5, 0.02, 'Time (s)', ha='center', fontsize=12)
        fig.text(0.02, 0.5, 'ΔF/F', va='center', rotation='vertical', fontsize=12)

        plt.suptitle(f'{self.annotation.video_name} - All Traces', fontsize=14)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()
        return fig

    def plot_raster(self, save_path: str = None):
        """Generate raster plot of all spikes"""
        fig, ax = plt.subplots(figsize=(14, 8))

        confidence_colors = {
            SpikeConfidence.CERTAIN.value: 'black',
            SpikeConfidence.PROBABLE.value: 'gray',
            SpikeConfidence.UNCERTAIN.value: 'lightgray'
        }

        for roi in self.annotation.rois:
            spikes = self.annotation.get_spikes_for_roi(roi.roi_id)
            for spike in spikes:
                color = confidence_colors.get(spike.confidence, 'black')
                ax.scatter(spike.time_seconds, roi.roi_id, c=color, s=20, marker='|')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('ROI ID', fontsize=12)
        ax.set_title(f'{self.annotation.video_name} - Spike Raster\n'
                    f'({len(self.annotation.spikes)} total spikes)', fontsize=14)

        # Legend
        for conf, color in confidence_colors.items():
            ax.scatter([], [], c=color, s=50, marker='|', label=conf)
        ax.legend(title='Confidence', loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()
        return fig

    def plot_activity_heatmap(self, save_path: str = None):
        """Generate activity heatmap"""
        # Create matrix of traces
        n_rois = len(self.annotation.rois)
        trace_matrix = np.zeros((n_rois, self.n_frames))

        for idx, roi in enumerate(self.annotation.rois):
            trace = self.traces[roi.roi_id]
            # Normalize
            trace_norm = (trace - trace.min()) / (trace.max() - trace.min() + 1e-10)
            trace_matrix[idx] = trace_norm

        fig, ax = plt.subplots(figsize=(14, 8))

        time = np.arange(self.n_frames) / self.annotation.fps
        im = ax.imshow(trace_matrix, aspect='auto', cmap='RdYlBu_r',
                      extent=[time[0], time[-1], n_rois - 0.5, -0.5])

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('ROI ID', fontsize=12)
        ax.set_title(f'{self.annotation.video_name} - Activity Heatmap', fontsize=14)

        plt.colorbar(im, ax=ax, label='Normalized ΔF/F')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()
        return fig

    def plot_single_roi_detail(self, roi_id: int, save_path: str = None):
        """Detailed view of a single ROI"""
        roi = self.annotation.get_roi_by_id(roi_id)
        if roi is None:
            print(f"ROI {roi_id} not found")
            return

        spikes = self.annotation.get_spikes_for_roi(roi_id)
        trace = self.traces[roi_id]
        time = np.arange(self.n_frames) / self.annotation.fps

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.5])

        # ROI location
        ax_roi = fig.add_subplot(gs[0, 0])
        ax_roi.imshow(self.std_proj, cmap='gray',
                     vmin=np.percentile(self.std_proj, 1),
                     vmax=np.percentile(self.std_proj, 99))
        polygon = MplPolygon(roi.polygon, fill=False, edgecolor='cyan', linewidth=2)
        ax_roi.add_patch(polygon)
        ax_roi.set_xlim(roi.bbox[0] - 20, roi.bbox[2] + 20)
        ax_roi.set_ylim(roi.bbox[3] + 20, roi.bbox[1] - 20)
        ax_roi.set_title(f'ROI {roi_id} Location')

        # Full trace
        ax_trace = fig.add_subplot(gs[0, 1:])
        ax_trace.plot(time, trace, 'k-', linewidth=0.5)
        for spike in spikes:
            ax_trace.axvline(spike.time_seconds, color='red', alpha=0.5, linewidth=1)
        ax_trace.set_xlabel('Time (s)')
        ax_trace.set_ylabel('ΔF/F')
        ax_trace.set_title(f'Full Trace ({len(spikes)} spikes)')

        # Spike-triggered average
        ax_sta = fig.add_subplot(gs[1, 0])
        if spikes:
            window_frames = int(0.05 * self.annotation.fps)  # 50ms window
            snippets = []
            for spike in spikes:
                idx = spike.frame_index
                if idx >= window_frames and idx < self.n_frames - window_frames:
                    snippet = trace[idx - window_frames:idx + window_frames]
                    snippets.append(snippet)

            if snippets:
                snippets = np.array(snippets)
                mean_snippet = np.mean(snippets, axis=0)
                std_snippet = np.std(snippets, axis=0)
                t_snippet = np.arange(-window_frames, window_frames) / self.annotation.fps * 1000

                ax_sta.fill_between(t_snippet, mean_snippet - std_snippet,
                                   mean_snippet + std_snippet, alpha=0.3)
                ax_sta.plot(t_snippet, mean_snippet, 'k-', linewidth=2)
                ax_sta.axvline(0, color='red', linestyle='--', alpha=0.5)

        ax_sta.set_xlabel('Time from spike (ms)')
        ax_sta.set_ylabel('ΔF/F')
        ax_sta.set_title('Spike-Triggered Average')

        # ISI histogram
        ax_isi = fig.add_subplot(gs[1, 1])
        if len(spikes) > 1:
            spike_times = np.array([s.time_seconds for s in spikes])
            isis = np.diff(spike_times) * 1000  # Convert to ms
            ax_isi.hist(isis, bins=30, color='steelblue', edgecolor='black')
        ax_isi.set_xlabel('Inter-spike interval (ms)')
        ax_isi.set_ylabel('Count')
        ax_isi.set_title('ISI Distribution')

        # Amplitude distribution
        ax_amp = fig.add_subplot(gs[1, 2])
        if spikes:
            amplitudes = [s.amplitude for s in spikes if s.amplitude is not None]
            if amplitudes:
                ax_amp.hist(amplitudes, bins=20, color='coral', edgecolor='black')
        ax_amp.set_xlabel('Spike amplitude (ΔF/F)')
        ax_amp.set_ylabel('Count')
        ax_amp.set_title('Amplitude Distribution')

        # Info table
        ax_info = fig.add_subplot(gs[2, :])
        ax_info.axis('off')
        info_text = (
            f"ROI ID: {roi_id}  |  Quality: {roi.quality}  |  Shape: {roi.shape}  |  "
            f"Area: {roi.area_pixels:.0f} px\n"
            f"Spikes: {len(spikes)}  |  "
            f"Spike rate: {len(spikes) / (self.n_frames / self.annotation.fps):.2f} Hz  |  "
            f"Mean amplitude: {np.mean([s.amplitude for s in spikes if s.amplitude]):.4f}" if spikes else "No spikes"
        )
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center',
                    fontsize=12, transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.suptitle(f'{self.annotation.video_name} - ROI {roi_id} Detail', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()
        return fig

    def generate_summary_report(self, output_dir: str = "annotation_report"):
        """Generate complete summary report with all figures"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating summary report in {output_dir}/")
        print("=" * 50)

        # Generate all figures
        print("1. ROI Overview...")
        self.plot_roi_overview(save_path=output_path / "01_roi_overview.png")

        print("2. All Traces...")
        self.plot_all_traces(save_path=output_path / "02_all_traces.png")

        print("3. Raster Plot...")
        self.plot_raster(save_path=output_path / "03_raster.png")

        print("4. Activity Heatmap...")
        self.plot_activity_heatmap(save_path=output_path / "04_heatmap.png")

        # Generate detail for most active ROIs
        print("5. Individual ROI Details...")
        spike_counts = []
        for roi in self.annotation.rois:
            n_spikes = len(self.annotation.get_spikes_for_roi(roi.roi_id))
            spike_counts.append((roi.roi_id, n_spikes))

        spike_counts.sort(key=lambda x: x[1], reverse=True)
        top_rois = spike_counts[:5]  # Top 5 most active

        for roi_id, n_spikes in top_rois:
            if n_spikes > 0:
                self.plot_single_roi_detail(
                    roi_id,
                    save_path=output_path / f"05_roi_{roi_id}_detail.png"
                )

        # Export statistics
        print("6. Exporting statistics...")
        self.export_statistics(output_path / "statistics.csv")

        # Print summary
        print("\n" + "=" * 50)
        print(self.annotation.summary())
        print("=" * 50)
        print(f"\nReport saved to {output_dir}/")

    def export_statistics(self, output_path: str = "statistics.csv"):
        """Export annotation statistics to CSV"""
        stats = []

        for roi in self.annotation.rois:
            spikes = self.annotation.get_spikes_for_roi(roi.roi_id)
            trace = self.traces[roi.roi_id]

            duration = self.n_frames / self.annotation.fps

            stat = {
                'roi_id': roi.roi_id,
                'quality': roi.quality,
                'shape': roi.shape,
                'area_pixels': roi.area_pixels,
                'centroid_x': roi.centroid[0],
                'centroid_y': roi.centroid[1],
                'n_spikes': len(spikes),
                'spike_rate_hz': len(spikes) / duration,
                'mean_amplitude': np.mean([s.amplitude for s in spikes]) if spikes else None,
                'std_amplitude': np.std([s.amplitude for s in spikes]) if spikes else None,
                'trace_mean': np.mean(trace),
                'trace_std': np.std(trace),
                'n_certain_spikes': sum(1 for s in spikes if s.confidence == SpikeConfidence.CERTAIN.value),
                'n_probable_spikes': sum(1 for s in spikes if s.confidence == SpikeConfidence.PROBABLE.value),
                'n_uncertain_spikes': sum(1 for s in spikes if s.confidence == SpikeConfidence.UNCERTAIN.value),
            }
            stats.append(stat)

        df = pd.DataFrame(stats)
        df.to_csv(output_path, index=False)
        print(f"Statistics exported to {output_path}")
        return df


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) > 1:
        annotation_path = sys.argv[1]
    else:
        # Look for annotation files
        annotation_files = list(Path("annotations").glob("*_annotations.json"))
        if annotation_files:
            annotation_path = str(annotation_files[0])
            print(f"Found annotation file: {annotation_path}")
        else:
            print("Usage: python annotation_viewer.py <annotation_path> [video_path]")
            print("No annotation files found in annotations/")
            return

    video_path = sys.argv[2] if len(sys.argv) > 2 else None

    viewer = AnnotationViewer(annotation_path, video_path)

    # Generate full report
    viewer.generate_summary_report()


if __name__ == "__main__":
    main()
