"""
Frequency Analysis Tool for Voltage Imaging Traces
Analyzes spectral content of ROI traces to inform filtering decisions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from matplotlib.path import Path as MplPath
import tifffile
from pathlib import Path

from annotation_schema import VideoAnnotation


class FrequencyAnalyzer:
    """Analyze frequency content of voltage imaging traces"""

    def __init__(self, video_path: str, annotation_path: str = None, fps: float = 200):
        """
        Initialize analyzer

        Parameters:
        -----------
        video_path : str
            Path to TIFF video
        annotation_path : str
            Path to ROI annotations JSON (optional, will look for default)
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
        print(f"Loaded: {self.n_frames} frames, {self.height}x{self.width}, {fps} Hz")

        # Load annotations
        if annotation_path is None:
            annotation_path = f"annotations/{self.video_name}_rois.json"

        print(f"Loading annotations: {annotation_path}")
        self.annotation = VideoAnnotation.load(annotation_path)
        print(f"Loaded {len(self.annotation.rois)} ROIs")

        # Extract traces
        self._extract_traces()

    def _extract_traces(self):
        """Extract traces for all ROIs"""
        print("Extracting traces...")
        self.raw_traces = {}
        self.dff_traces = {}

        for roi in self.annotation.rois:
            # Create mask from polygon
            polygon = np.array(roi.polygon)
            y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
            points = np.vstack((x_coords.ravel(), y_coords.ravel())).T
            path = MplPath(polygon)
            mask = path.contains_points(points).reshape(self.height, self.width)

            # Extract trace
            trace = np.array([np.mean(self.video[t][mask]) for t in range(self.n_frames)])
            self.raw_traces[roi.roi_id] = trace

            # Compute dF/F
            f0 = np.percentile(trace, 10)
            self.dff_traces[roi.roi_id] = (trace - f0) / f0

        print(f"Extracted {len(self.raw_traces)} traces")

    def analyze_single_roi(self, roi_id: int, save_path: str = None):
        """Analyze frequency content of a single ROI"""
        trace = self.dff_traces[roi_id]
        n = len(trace)

        # Compute FFT
        yf = fft(trace)
        xf = fftfreq(n, 1/self.fps)[:n//2]
        power = 2.0/n * np.abs(yf[0:n//2])

        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(trace, self.fps, nperseg=min(256, n//4))

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Time domain
        time = np.arange(n) / self.fps
        axes[0, 0].plot(time, trace, 'k-', linewidth=0.5)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('ΔF/F')
        axes[0, 0].set_title(f'ROI {roi_id} - Time Domain')

        # Power spectrum (log scale)
        axes[0, 1].semilogy(xf, power)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power (log)')
        axes[0, 1].set_title('Power Spectrum (log scale)')
        axes[0, 1].set_xlim(0, self.fps/2)

        # Power spectrum (zoomed, linear)
        axes[1, 0].plot(xf, power, 'b-', linewidth=1)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].set_title('Power Spectrum (0-100 Hz)')
        axes[1, 0].set_xlim(0, 100)

        # Mark common frequencies
        axes[1, 0].axvline(10, color='g', linestyle='--', alpha=0.5, label='10 Hz')
        axes[1, 0].axvline(50, color='r', linestyle='--', alpha=0.5, label='50 Hz (line noise)')
        axes[1, 0].axvline(60, color='orange', linestyle='--', alpha=0.5, label='60 Hz (line noise)')
        axes[1, 0].legend()

        # Spectrogram
        if Sxx.size > 0:
            im = axes[1, 1].pcolormesh(t, f, 10*np.log10(Sxx + 1e-10),
                                        shading='gouraud', cmap='viridis')
            axes[1, 1].set_ylabel('Frequency (Hz)')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_title('Spectrogram')
            axes[1, 1].set_ylim(0, 100)
            plt.colorbar(im, ax=axes[1, 1], label='Power (dB)')

        plt.suptitle(f'{self.video_name} - ROI {roi_id} Frequency Analysis', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

        # Print summary
        print(f"\n=== ROI {roi_id} Frequency Analysis ===")
        print(f"Duration: {n/self.fps:.1f} s")
        print(f"Nyquist frequency: {self.fps/2:.1f} Hz")

        # Find dominant frequencies
        print(f"\nDominant frequencies (top 5):")
        # Exclude DC component (0 Hz)
        power_no_dc = power.copy()
        power_no_dc[0] = 0
        top_idx = np.argsort(power_no_dc)[-5:][::-1]
        for idx in top_idx:
            print(f"  {xf[idx]:.1f} Hz: power = {power[idx]:.6f}")

        # Check for line noise
        idx_50 = np.argmin(np.abs(xf - 50))
        idx_60 = np.argmin(np.abs(xf - 60))
        noise_floor = np.median(power[xf > 70])

        if power[idx_50] > 3 * noise_floor:
            print(f"\n⚠️  50 Hz line noise detected (power: {power[idx_50]:.6f})")
        if power[idx_60] > 3 * noise_floor:
            print(f"\n⚠️  60 Hz line noise detected (power: {power[idx_60]:.6f})")

        # Suggest filter cutoff
        # Find where power drops to noise floor
        cumsum = np.cumsum(power_no_dc)
        total_power = cumsum[-1]
        idx_95 = np.searchsorted(cumsum, 0.95 * total_power)
        suggested_cutoff = xf[idx_95] if idx_95 < len(xf) else self.fps/2

        print(f"\n📊 Suggested lowpass cutoff: {suggested_cutoff:.0f} Hz")
        print(f"   (95% of signal power below this frequency)")

        return xf, power

    def analyze_all_rois(self, output_dir: str = "frequency_analysis"):
        """Analyze all ROIs and create summary"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_powers = []
        all_freqs = None
        suggested_cutoffs = []

        print(f"\nAnalyzing {len(self.annotation.rois)} ROIs...")
        print("=" * 50)

        for roi in self.annotation.rois:
            roi_id = roi.roi_id
            trace = self.dff_traces[roi_id]
            n = len(trace)

            # Compute FFT
            yf = fft(trace)
            xf = fftfreq(n, 1/self.fps)[:n//2]
            power = 2.0/n * np.abs(yf[0:n//2])

            all_powers.append(power)
            if all_freqs is None:
                all_freqs = xf

            # Find suggested cutoff
            power_no_dc = power.copy()
            power_no_dc[0] = 0
            cumsum = np.cumsum(power_no_dc)
            total_power = cumsum[-1]
            idx_95 = np.searchsorted(cumsum, 0.95 * total_power)
            cutoff = xf[idx_95] if idx_95 < len(xf) else self.fps/2
            suggested_cutoffs.append(cutoff)

            # Save individual plot
            self._save_single_spectrum(roi_id, xf, power, output_path / f"roi_{roi_id}_spectrum.png")

        # Create summary figure
        self._create_summary_figure(all_freqs, all_powers, suggested_cutoffs, output_path)

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Mean suggested cutoff: {np.mean(suggested_cutoffs):.1f} Hz")
        print(f"Median suggested cutoff: {np.median(suggested_cutoffs):.1f} Hz")
        print(f"Range: {np.min(suggested_cutoffs):.1f} - {np.max(suggested_cutoffs):.1f} Hz")
        print(f"\n💡 Recommended lowpass filter: {np.median(suggested_cutoffs):.0f} Hz")
        print(f"\nResults saved to {output_dir}/")

        return all_freqs, all_powers, suggested_cutoffs

    def _save_single_spectrum(self, roi_id, xf, power, save_path):
        """Save a simple spectrum plot"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(xf, power, 'b-', linewidth=0.8)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f'ROI {roi_id} Power Spectrum')
        ax.set_xlim(0, 100)
        ax.axvline(50, color='r', linestyle='--', alpha=0.3)
        ax.axvline(60, color='orange', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    def _create_summary_figure(self, freqs, all_powers, cutoffs, output_path):
        """Create summary figure for all ROIs"""
        all_powers = np.array(all_powers)
        mean_power = np.mean(all_powers, axis=0)
        std_power = np.std(all_powers, axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Mean power spectrum
        axes[0, 0].semilogy(freqs, mean_power, 'b-', linewidth=2)
        axes[0, 0].fill_between(freqs, mean_power - std_power, mean_power + std_power,
                                 alpha=0.3)
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Power (log)')
        axes[0, 0].set_title('Mean Power Spectrum (all ROIs)')
        axes[0, 0].set_xlim(0, self.fps/2)

        # Zoomed mean spectrum
        axes[0, 1].plot(freqs, mean_power, 'b-', linewidth=2)
        axes[0, 1].fill_between(freqs, mean_power - std_power, mean_power + std_power,
                                 alpha=0.3)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].set_title('Mean Power Spectrum (0-100 Hz)')
        axes[0, 1].set_xlim(0, 100)
        axes[0, 1].axvline(50, color='r', linestyle='--', alpha=0.5, label='50 Hz')
        axes[0, 1].axvline(60, color='orange', linestyle='--', alpha=0.5, label='60 Hz')
        axes[0, 1].axvline(np.median(cutoffs), color='g', linestyle='-', alpha=0.7,
                          label=f'Suggested cutoff ({np.median(cutoffs):.0f} Hz)')
        axes[0, 1].legend()

        # Heatmap of all spectra
        power_matrix = np.log10(all_powers + 1e-10)
        im = axes[1, 0].imshow(power_matrix, aspect='auto', cmap='viridis',
                               extent=[freqs[0], freqs[-1], len(all_powers)-0.5, -0.5])
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('ROI')
        axes[1, 0].set_title('Power Spectra Heatmap (all ROIs)')
        axes[1, 0].set_xlim(0, 100)
        plt.colorbar(im, ax=axes[1, 0], label='log10(Power)')

        # Histogram of suggested cutoffs
        axes[1, 1].hist(cutoffs, bins=20, color='steelblue', edgecolor='black')
        axes[1, 1].axvline(np.median(cutoffs), color='r', linestyle='--',
                          label=f'Median: {np.median(cutoffs):.1f} Hz')
        axes[1, 1].set_xlabel('Suggested Cutoff Frequency (Hz)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Suggested Cutoffs')
        axes[1, 1].legend()

        plt.suptitle(f'{self.video_name} - Frequency Analysis Summary', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / 'summary.png', dpi=150, bbox_inches='tight')
        plt.show()

    def compare_filters(self, roi_id: int, cutoffs: list = [30, 50, 80]):
        """Compare different lowpass filter cutoffs on a single ROI"""
        trace = self.dff_traces[roi_id]
        time = np.arange(len(trace)) / self.fps

        fig, axes = plt.subplots(len(cutoffs) + 2, 1, figsize=(14, 3 * (len(cutoffs) + 2)),
                                 sharex=True)

        # Original trace
        axes[0].plot(time, trace, 'k-', linewidth=0.5)
        axes[0].set_ylabel('ΔF/F')
        axes[0].set_title(f'ROI {roi_id} - Original')

        # Filtered traces
        for i, cutoff in enumerate(cutoffs):
            sos = signal.butter(4, cutoff, btype='low', fs=self.fps, output='sos')
            filtered = signal.sosfiltfilt(sos, trace)

            axes[i + 1].plot(time, filtered, 'b-', linewidth=0.5)
            axes[i + 1].set_ylabel('ΔF/F')
            axes[i + 1].set_title(f'Lowpass {cutoff} Hz')

        # Savitzky-Golay filter for comparison
        sg_filtered = signal.savgol_filter(trace, window_length=5, polyorder=2)
        axes[-1].plot(time, sg_filtered, 'g-', linewidth=0.5)
        axes[-1].set_ylabel('ΔF/F')
        axes[-1].set_title('Savitzky-Golay (window=5, poly=2)')

        axes[-1].set_xlabel('Time (s)')

        plt.suptitle(f'{self.video_name} - Filter Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'filter_comparison_roi_{roi_id}.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main entry point"""
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "Voltron2-ST_fish2_fov25_spin_conf-5ms_fTL_5_13_25_027_crop_cleaned.tif"

    annotation_path = sys.argv[2] if len(sys.argv) > 2 else None
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 200

    analyzer = FrequencyAnalyzer(video_path, annotation_path, fps)

    # Create output directory
    output_dir = Path("frequency_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze EACH ROI with full power spectrum analysis
    print("\n" + "=" * 50)
    print("POWER SPECTRUM ANALYSIS FOR EACH ROI")
    print("=" * 50)

    all_cutoffs = []
    for roi in analyzer.annotation.rois:
        roi_id = roi.roi_id
        print(f"\n--- ROI {roi_id} ---")
        xf, power = analyzer.analyze_single_roi(
            roi_id,
            save_path=str(output_dir / f"roi_{roi_id}_full_analysis.png")
        )

        # Collect suggested cutoff
        power_no_dc = power.copy()
        power_no_dc[0] = 0
        cumsum = np.cumsum(power_no_dc)
        idx_95 = np.searchsorted(cumsum, 0.95 * cumsum[-1])
        cutoff = xf[idx_95] if idx_95 < len(xf) else fps/2
        all_cutoffs.append(cutoff)

    # Print overall summary
    print("\n" + "=" * 50)
    print("OVERALL SUMMARY")
    print("=" * 50)
    print(f"Analyzed {len(analyzer.annotation.rois)} ROIs")
    print(f"Mean suggested cutoff: {np.mean(all_cutoffs):.1f} Hz")
    print(f"Median suggested cutoff: {np.median(all_cutoffs):.1f} Hz")
    print(f"Range: {np.min(all_cutoffs):.1f} - {np.max(all_cutoffs):.1f} Hz")
    print(f"\nRecommended lowpass filter: {np.median(all_cutoffs):.0f} Hz")
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
