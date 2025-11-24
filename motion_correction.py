"""
Motion Correction for Voltage Imaging
Rigid and non-rigid registration to correct for sample movement
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift
from skimage.registration import phase_cross_correlation
from pathlib import Path
import tifffile
import json
from typing import Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class MotionCorrectionResult:
    """Results from motion correction"""
    shifts_x: list  # X shifts per frame
    shifts_y: list  # Y shifts per frame
    max_shift: float  # Maximum shift detected
    mean_shift: float  # Mean shift magnitude
    correlation_pre: list  # Frame-to-reference correlation before correction
    correlation_post: list  # Frame-to-reference correlation after correction
    reference_frame: int  # Which frame was used as reference
    method: str  # 'rigid' or 'nonrigid'

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class MotionCorrector:
    """Motion correction for voltage imaging data"""

    def __init__(self, video_path: str = None, video: np.ndarray = None):
        """
        Initialize motion corrector

        Parameters:
        -----------
        video_path : str
            Path to TIFF video file
        video : np.ndarray
            Video array directly (if already loaded)
        """
        if video is not None:
            self.video = video
            self.video_path = None
            self.video_name = "video"
        elif video_path is not None:
            self.video_path = video_path
            self.video_name = Path(video_path).stem
            print(f"Loading video: {video_path}")
            self.video = tifffile.imread(video_path)
        else:
            raise ValueError("Provide either video_path or video array")

        self.n_frames, self.height, self.width = self.video.shape
        print(f"Video shape: {self.n_frames} frames, {self.height}x{self.width}")

        self.corrected_video = None
        self.result = None

    def _phase_correlation(self, ref: np.ndarray, target: np.ndarray, debug: bool = False) -> Tuple[float, float]:
        """
        Compute shift between two images using phase correlation

        Returns:
        --------
        (shift_y, shift_x) : tuple
            Shift needed to align target to reference
        """
        # phase_cross_correlation returns (shift, error, phase_diff)
        # shift is (y, x) to move target to match reference
        # upsample_factor=10 gives subpixel precision
        shift, error, _ = phase_cross_correlation(ref, target, upsample_factor=10)

        if debug:
            print(f"  DEBUG: shift={shift}, error={error}")

        return shift[0], shift[1]

    def _subpixel_shift(self, ref: np.ndarray, target: np.ndarray,
                        initial_shift: Tuple[float, float]) -> Tuple[float, float]:
        """
        Refine shift estimate to subpixel precision using local quadratic fit
        """
        # Apply initial shift
        shifted = ndimage.shift(target, initial_shift, order=1, mode='constant')

        # Try small offsets around initial estimate
        best_corr = np.corrcoef(ref.ravel(), shifted.ravel())[0, 1]
        best_shift = initial_shift

        for dy in [-0.5, 0, 0.5]:
            for dx in [-0.5, 0, 0.5]:
                test_shift = (initial_shift[0] + dy, initial_shift[1] + dx)
                test_shifted = ndimage.shift(target, test_shift, order=1, mode='constant')
                corr = np.corrcoef(ref.ravel(), test_shifted.ravel())[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_shift = test_shift

        return best_shift

    def _compute_correlation(self, ref: np.ndarray, target: np.ndarray) -> float:
        """Compute correlation between two frames"""
        return np.corrcoef(ref.ravel(), target.ravel())[0, 1]

    def estimate_reference_frame(self, method: str = 'mean') -> int:
        """
        Estimate best reference frame

        Parameters:
        -----------
        method : str
            'mean' - use mean projection (virtual frame)
            'middle' - use middle frame
            'sharpest' - use frame with highest variance (sharpest)
            'most_correlated' - use frame most correlated with others
        """
        if method == 'mean':
            return -1  # Will use mean projection
        elif method == 'middle':
            return self.n_frames // 2
        elif method == 'sharpest':
            # Frame with highest variance/gradient
            sharpness = []
            for i in range(0, self.n_frames, max(1, self.n_frames // 100)):
                frame = self.video[i].astype(float)
                # Laplacian variance as sharpness metric
                lap = ndimage.laplace(frame)
                sharpness.append((i, np.var(lap)))
            return max(sharpness, key=lambda x: x[1])[0]
        elif method == 'most_correlated':
            # Sample frames and find one most correlated with others
            sample_idx = np.linspace(0, self.n_frames-1, min(50, self.n_frames)).astype(int)
            sample_frames = self.video[sample_idx]

            mean_corrs = []
            for i, frame in enumerate(sample_frames):
                corrs = [self._compute_correlation(frame, other)
                         for j, other in enumerate(sample_frames) if i != j]
                mean_corrs.append(np.mean(corrs))

            return sample_idx[np.argmax(mean_corrs)]
        else:
            raise ValueError(f"Unknown method: {method}")

    def correct_rigid(self, reference_method: str = 'mean',
                      max_shift: int = 50,
                      subpixel: bool = True,
                      verbose: bool = True) -> np.ndarray:
        """
        Apply rigid (translation-only) motion correction

        Parameters:
        -----------
        reference_method : str
            How to compute reference ('mean', 'middle', 'sharpest', 'most_correlated')
        max_shift : int
            Maximum allowed shift in pixels (larger shifts are clipped)
        subpixel : bool
            Whether to estimate subpixel shifts
        verbose : bool
            Print progress

        Returns:
        --------
        corrected_video : np.ndarray
            Motion-corrected video
        """
        if verbose:
            print(f"\nRigid Motion Correction")
            print(f"Reference method: {reference_method}")
            print("=" * 50)

        # Get reference frame
        ref_idx = self.estimate_reference_frame(reference_method)
        if ref_idx == -1:
            reference = np.mean(self.video, axis=0)
            if verbose:
                print("Using mean projection as reference")
        else:
            reference = self.video[ref_idx].astype(float)
            if verbose:
                print(f"Using frame {ref_idx} as reference")

        reference = reference.astype(float)

        # Initialize output
        self.corrected_video = np.zeros_like(self.video)
        shifts_x = []
        shifts_y = []
        corr_pre = []
        corr_post = []

        if verbose:
            print(f"\nProcessing {self.n_frames} frames...")
            print(f"Reference shape: {reference.shape}, range: {reference.min():.1f} - {reference.max():.1f}")

            # Check if there's actual motion by comparing first and last frames
            first_frame = self.video[0].astype(float)
            last_frame = self.video[-1].astype(float)
            mid_frame = self.video[self.n_frames//2].astype(float)
            print(f"Mean abs diff (frame 0 vs mid): {np.abs(first_frame - mid_frame).mean():.2f}")
            print(f"Mean abs diff (frame 0 vs last): {np.abs(first_frame - last_frame).mean():.2f}")

            # Test phase correlation directly with known shifted image
            test_shift = ndimage.shift(first_frame, (5, 3), order=1)
            test_result, _, _ = phase_cross_correlation(first_frame, test_shift, upsample_factor=10)
            print(f"Self-test: shifted by (5,3), detected: ({test_result[0]:.2f}, {test_result[1]:.2f})")

        for i in range(self.n_frames):
            frame = self.video[i].astype(float)

            # Compute correlation before correction
            corr_pre.append(self._compute_correlation(reference, frame))

            # Estimate shift (already subpixel via upsample_factor in phase_cross_correlation)
            debug_this = verbose and i < 5
            shift_y, shift_x = self._phase_correlation(reference, frame, debug=debug_this)

            if debug_this:
                print(f"  Frame {i}: shift=({shift_y:.3f}, {shift_x:.3f})")

            # Clip to max shift
            shift_y = np.clip(shift_y, -max_shift, max_shift)
            shift_x = np.clip(shift_x, -max_shift, max_shift)

            shifts_y.append(float(shift_y))
            shifts_x.append(float(shift_x))

            # Apply correction
            corrected = ndimage.shift(frame, (shift_y, shift_x), order=1, mode='constant')
            self.corrected_video[i] = corrected.astype(self.video.dtype)

            # Compute correlation after correction
            corr_post.append(self._compute_correlation(reference, corrected))

            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{self.n_frames} frames")

        # Store results
        shifts_magnitude = np.sqrt(np.array(shifts_x)**2 + np.array(shifts_y)**2)
        self.result = MotionCorrectionResult(
            shifts_x=shifts_x,
            shifts_y=shifts_y,
            max_shift=float(np.max(shifts_magnitude)),
            mean_shift=float(np.mean(shifts_magnitude)),
            correlation_pre=corr_pre,
            correlation_post=corr_post,
            reference_frame=int(ref_idx),
            method='rigid'
        )

        if verbose:
            print(f"\nDone!")
            print(f"Max shift: {self.result.max_shift:.2f} pixels")
            print(f"Mean shift: {self.result.mean_shift:.2f} pixels")
            print(f"Correlation improved: {np.mean(corr_pre):.4f} → {np.mean(corr_post):.4f}")

        return self.corrected_video

    def correct_nonrigid(self, grid_size: Tuple[int, int] = (4, 4),
                         reference_method: str = 'mean',
                         max_shift: int = 20,
                         verbose: bool = True) -> np.ndarray:
        """
        Apply non-rigid motion correction using local patch registration

        Parameters:
        -----------
        grid_size : tuple
            Number of patches in (y, x) directions
        reference_method : str
            How to compute reference
        max_shift : int
            Maximum allowed shift per patch
        verbose : bool
            Print progress

        Returns:
        --------
        corrected_video : np.ndarray
            Motion-corrected video
        """
        if verbose:
            print(f"\nNon-Rigid Motion Correction")
            print(f"Grid size: {grid_size}")
            print("=" * 50)

        # Get reference
        ref_idx = self.estimate_reference_frame(reference_method)
        if ref_idx == -1:
            reference = np.mean(self.video, axis=0)
        else:
            reference = self.video[ref_idx].astype(float)

        reference = reference.astype(float)

        # Compute patch sizes
        patch_h = self.height // grid_size[0]
        patch_w = self.width // grid_size[1]

        # Initialize output
        self.corrected_video = np.zeros_like(self.video)
        all_shifts_x = []
        all_shifts_y = []
        corr_pre = []
        corr_post = []

        if verbose:
            print(f"Patch size: {patch_h}x{patch_w} pixels")
            print(f"\nProcessing {self.n_frames} frames...")

        for i in range(self.n_frames):
            frame = self.video[i].astype(float)
            corr_pre.append(self._compute_correlation(reference, frame))

            # Compute shifts for each patch
            shift_field_y = np.zeros(grid_size)
            shift_field_x = np.zeros(grid_size)

            for gy in range(grid_size[0]):
                for gx in range(grid_size[1]):
                    # Extract patches
                    y_start = gy * patch_h
                    y_end = min((gy + 1) * patch_h, self.height)
                    x_start = gx * patch_w
                    x_end = min((gx + 1) * patch_w, self.width)

                    ref_patch = reference[y_start:y_end, x_start:x_end]
                    frame_patch = frame[y_start:y_end, x_start:x_end]

                    # Phase correlation on patch
                    f_ref = fft2(ref_patch)
                    f_target = fft2(frame_patch)
                    cross_power = (f_ref * np.conj(f_target)) / (np.abs(f_ref * np.conj(f_target)) + 1e-10)
                    correlation = np.real(ifft2(cross_power))
                    correlation = fftshift(correlation)

                    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
                    shift_y = max_idx[0] - ref_patch.shape[0] // 2
                    shift_x = max_idx[1] - ref_patch.shape[1] // 2

                    # Clip shifts
                    shift_field_y[gy, gx] = np.clip(shift_y, -max_shift, max_shift)
                    shift_field_x[gy, gx] = np.clip(shift_x, -max_shift, max_shift)

            # Interpolate shift field to full resolution
            from scipy.interpolate import RectBivariateSpline

            grid_y = np.linspace(patch_h/2, self.height - patch_h/2, grid_size[0])
            grid_x = np.linspace(patch_w/2, self.width - patch_w/2, grid_size[1])

            interp_y = RectBivariateSpline(grid_y, grid_x, shift_field_y, kx=1, ky=1)
            interp_x = RectBivariateSpline(grid_y, grid_x, shift_field_x, kx=1, ky=1)

            full_y = np.arange(self.height)
            full_x = np.arange(self.width)
            shift_map_y = interp_y(full_y, full_x)
            shift_map_x = interp_x(full_y, full_x)

            # Apply non-rigid correction using coordinate mapping
            coords_y, coords_x = np.meshgrid(full_y, full_x, indexing='ij')
            new_coords_y = coords_y + shift_map_y
            new_coords_x = coords_x + shift_map_x

            corrected = ndimage.map_coordinates(frame, [new_coords_y, new_coords_x],
                                                order=1, mode='constant')
            self.corrected_video[i] = corrected.astype(self.video.dtype)

            corr_post.append(self._compute_correlation(reference, corrected))

            # Store mean shifts for this frame
            all_shifts_y.append(float(np.mean(shift_field_y)))
            all_shifts_x.append(float(np.mean(shift_field_x)))

            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{self.n_frames} frames")

        # Store results
        shifts_magnitude = np.sqrt(np.array(all_shifts_x)**2 + np.array(all_shifts_y)**2)
        self.result = MotionCorrectionResult(
            shifts_x=all_shifts_x,
            shifts_y=all_shifts_y,
            max_shift=float(np.max(shifts_magnitude)),
            mean_shift=float(np.mean(shifts_magnitude)),
            correlation_pre=corr_pre,
            correlation_post=corr_post,
            reference_frame=int(ref_idx),
            method='nonrigid'
        )

        if verbose:
            print(f"\nDone!")
            print(f"Mean shift: {self.result.mean_shift:.2f} pixels")
            print(f"Correlation improved: {np.mean(corr_pre):.4f} → {np.mean(corr_post):.4f}")

        return self.corrected_video

    def plot_diagnostics(self, save_path: str = None):
        """Plot motion correction diagnostics"""
        if self.result is None:
            print("Run correction first!")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Shifts over time
        axes[0, 0].plot(self.result.shifts_x, 'b-', linewidth=0.5, label='X shift')
        axes[0, 0].plot(self.result.shifts_y, 'r-', linewidth=0.5, label='Y shift')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Shift (pixels)')
        axes[0, 0].set_title('Motion Over Time')
        axes[0, 0].legend()

        # Shift magnitude
        magnitude = np.sqrt(np.array(self.result.shifts_x)**2 + np.array(self.result.shifts_y)**2)
        axes[0, 1].plot(magnitude, 'k-', linewidth=0.5)
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Shift magnitude (pixels)')
        axes[0, 1].set_title(f'Shift Magnitude (max: {self.result.max_shift:.2f} px)')

        # XY scatter
        axes[0, 2].scatter(self.result.shifts_x, self.result.shifts_y,
                          c=np.arange(len(self.result.shifts_x)), cmap='viridis', s=1)
        axes[0, 2].set_xlabel('X shift (pixels)')
        axes[0, 2].set_ylabel('Y shift (pixels)')
        axes[0, 2].set_title('XY Motion Path')
        axes[0, 2].axis('equal')

        # Correlation improvement
        axes[1, 0].plot(self.result.correlation_pre, 'r-', linewidth=0.5, alpha=0.7, label='Before')
        axes[1, 0].plot(self.result.correlation_post, 'g-', linewidth=0.5, alpha=0.7, label='After')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Correlation with reference')
        axes[1, 0].set_title('Frame-Reference Correlation')
        axes[1, 0].legend()

        # Correlation histogram
        axes[1, 1].hist(self.result.correlation_pre, bins=50, alpha=0.5, label='Before', color='red')
        axes[1, 1].hist(self.result.correlation_post, bins=50, alpha=0.5, label='After', color='green')
        axes[1, 1].set_xlabel('Correlation')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Correlation Distribution')
        axes[1, 1].legend()

        # Before/after comparison (mean projection)
        mean_before = np.mean(self.video, axis=0)
        mean_after = np.mean(self.corrected_video, axis=0)

        # Show difference in sharpness
        axes[1, 2].imshow(mean_after - mean_before, cmap='RdBu_r')
        axes[1, 2].set_title('Mean Projection Difference (After - Before)')
        axes[1, 2].axis('off')

        plt.suptitle(f'Motion Correction Diagnostics ({self.result.method})', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def plot_comparison(self, frame_idx: int = None, save_path: str = None):
        """Show before/after for a specific frame"""
        if self.corrected_video is None:
            print("Run correction first!")
            return

        if frame_idx is None:
            # Pick frame with largest shift
            magnitude = np.sqrt(np.array(self.result.shifts_x)**2 + np.array(self.result.shifts_y)**2)
            frame_idx = np.argmax(magnitude)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Before
        axes[0].imshow(self.video[frame_idx], cmap='gray')
        axes[0].set_title(f'Frame {frame_idx} - Before')
        axes[0].axis('off')

        # After
        axes[1].imshow(self.corrected_video[frame_idx], cmap='gray')
        axes[1].set_title(f'Frame {frame_idx} - After')
        axes[1].axis('off')

        # Difference
        diff = self.corrected_video[frame_idx].astype(float) - self.video[frame_idx].astype(float)
        axes[2].imshow(diff, cmap='RdBu_r')
        axes[2].set_title(f'Difference (shift: {self.result.shifts_x[frame_idx]:.1f}, {self.result.shifts_y[frame_idx]:.1f})')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def save_corrected(self, output_path: str = None):
        """Save corrected video and results"""
        if self.corrected_video is None:
            print("Run correction first!")
            return

        if output_path is None:
            output_path = f"{self.video_name}_corrected.tif"

        output_path = Path(output_path)

        # Save video
        tifffile.imwrite(str(output_path), self.corrected_video)
        print(f"Saved corrected video to: {output_path}")

        # Save motion correction results
        results_path = output_path.with_suffix('.motion.json')
        self.result.save(str(results_path))
        print(f"Saved motion data to: {results_path}")

        return str(output_path)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python motion_correction.py <video.tif>                    # Rigid correction")
        print("  python motion_correction.py <video.tif> --nonrigid         # Non-rigid correction")
        print("  python motion_correction.py <video.tif> --output out.tif   # Custom output")
        return

    video_path = sys.argv[1]
    method = 'rigid'
    output_path = None

    if '--nonrigid' in sys.argv:
        method = 'nonrigid'

    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        output_path = sys.argv[idx + 1]

    # Run correction
    corrector = MotionCorrector(video_path)

    if method == 'rigid':
        corrector.correct_rigid()
    else:
        corrector.correct_nonrigid()

    # Plot diagnostics
    corrector.plot_diagnostics(save_path='motion_correction_diagnostics.png')
    corrector.plot_comparison()

    # Save
    corrector.save_corrected(output_path)


if __name__ == "__main__":
    main()
