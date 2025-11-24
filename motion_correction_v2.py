"""
Motion Correction for Voltage Imaging - Version 2
Simple, robust implementation using OpenCV
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
import json
import cv2


def compute_shift_cv2(reference: np.ndarray, target: np.ndarray) -> tuple:
    """
    Compute shift between two images using OpenCV's phase correlation

    Returns:
    --------
    (shift_x, shift_y) : tuple of floats
    """
    # Convert to float32 for OpenCV
    ref = reference.astype(np.float32)
    tgt = target.astype(np.float32)

    # Use OpenCV's phaseCorrelate - returns (x, y) shift
    shift, response = cv2.phaseCorrelate(ref, tgt)

    return shift[0], shift[1], response  # x, y, confidence


def compute_shift_xcorr(reference: np.ndarray, target: np.ndarray, max_shift: int = 50) -> tuple:
    """
    Compute shift using normalized cross-correlation (more robust)

    Returns:
    --------
    (shift_x, shift_y, confidence) : tuple
    """
    # Convert to float
    ref = reference.astype(np.float64)
    tgt = target.astype(np.float64)

    # Normalize
    ref = (ref - ref.mean()) / (ref.std() + 1e-10)
    tgt = (tgt - tgt.mean()) / (tgt.std() + 1e-10)

    # Compute cross-correlation using FFT
    from scipy import fft

    f_ref = fft.fft2(ref)
    f_tgt = fft.fft2(tgt)

    # Cross-correlation
    xcorr = fft.ifft2(f_ref * np.conj(f_tgt))
    xcorr = np.real(fft.fftshift(xcorr))

    # Find peak within max_shift region
    h, w = xcorr.shape
    cy, cx = h // 2, w // 2

    # Crop to search region
    y_min = max(0, cy - max_shift)
    y_max = min(h, cy + max_shift)
    x_min = max(0, cx - max_shift)
    x_max = min(w, cx + max_shift)

    search_region = xcorr[y_min:y_max, x_min:x_max]

    # Find peak
    peak_idx = np.unravel_index(np.argmax(search_region), search_region.shape)

    # Convert to shift relative to center
    shift_y = peak_idx[0] + y_min - cy
    shift_x = peak_idx[1] + x_min - cx

    confidence = search_region[peak_idx]

    return shift_x, shift_y, confidence


def compute_shift_template(reference: np.ndarray, target: np.ndarray, max_shift: int = 50) -> tuple:
    """
    Compute shift using OpenCV template matching (most robust)

    Returns:
    --------
    (shift_x, shift_y, confidence) : tuple
    """
    # Convert to float32
    ref = reference.astype(np.float32)
    tgt = target.astype(np.float32)

    # Normalize
    ref = (ref - ref.mean()) / (ref.std() + 1e-10)
    tgt = (tgt - tgt.mean()) / (tgt.std() + 1e-10)

    h, w = ref.shape

    # Crop center of reference as template
    margin = max_shift
    template = ref[margin:h-margin, margin:w-margin]

    # Match template in target
    result = cv2.matchTemplate(tgt, template, cv2.TM_CCORR_NORMED)

    # Find best match
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # max_loc is (x, y) of top-left corner of match
    # Expected position if no shift is (margin, margin)
    shift_x = margin - max_loc[0]
    shift_y = margin - max_loc[1]

    return shift_x, shift_y, max_val


def apply_shift(image: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    """Apply shift to image using OpenCV warpAffine"""
    h, w = image.shape

    # Create translation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Apply transformation
    shifted = cv2.warpAffine(
        image.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return shifted


def motion_correct_video(video_path: str,
                         output_path: str = None,
                         method: str = 'template',
                         reference_method: str = 'first',
                         max_shift: int = 50,
                         verbose: bool = True):
    """
    Apply motion correction to a video

    Parameters:
    -----------
    video_path : str
        Path to input TIFF video
    output_path : str
        Path for output (default: input_corrected.tif)
    method : str
        'template' (most robust), 'xcorr', or 'phase'
    reference_method : str
        'first', 'mean', or 'middle'
    max_shift : int
        Maximum shift to search for (pixels)
    verbose : bool
        Print progress

    Returns:
    --------
    dict with corrected_video, shifts_x, shifts_y, etc.
    """
    # Load video
    if verbose:
        print(f"Loading video: {video_path}")
    video = tifffile.imread(video_path)
    n_frames, height, width = video.shape
    if verbose:
        print(f"Shape: {n_frames} frames, {height}x{width}")

    # Select reference
    if reference_method == 'first':
        reference = video[0].astype(np.float64)
        ref_idx = 0
    elif reference_method == 'mean':
        reference = np.mean(video, axis=0).astype(np.float64)
        ref_idx = -1
    elif reference_method == 'middle':
        ref_idx = n_frames // 2
        reference = video[ref_idx].astype(np.float64)
    else:
        raise ValueError(f"Unknown reference method: {reference_method}")

    if verbose:
        print(f"Reference: {reference_method} (frame {ref_idx})")
        print(f"Method: {method}")
        print(f"Max shift: {max_shift} pixels")

    # Select shift computation method
    if method == 'template':
        compute_shift = lambda ref, tgt: compute_shift_template(ref, tgt, max_shift)
    elif method == 'xcorr':
        compute_shift = lambda ref, tgt: compute_shift_xcorr(ref, tgt, max_shift)
    elif method == 'phase':
        compute_shift = lambda ref, tgt: compute_shift_cv2(ref, tgt)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Test that shift detection is working
    if verbose:
        print("\nRunning self-test...")
        test_frame = video[0].astype(np.float64)
        # Artificially shift by known amount
        test_shifted = apply_shift(test_frame, 5, 3)
        detected_x, detected_y, conf = compute_shift(test_frame, test_shifted)
        print(f"  Applied shift: (5, 3)")
        print(f"  Detected shift: ({detected_x:.2f}, {detected_y:.2f}), confidence: {conf:.4f}")

        if abs(detected_x - 5) > 1 or abs(detected_y - 3) > 1:
            print("  WARNING: Self-test failed! Shift detection may not work correctly.")
        else:
            print("  Self-test PASSED")

    # Process all frames
    if verbose:
        print(f"\nProcessing {n_frames} frames...")

    corrected_video = np.zeros_like(video)
    shifts_x = []
    shifts_y = []
    confidences = []

    for i in range(n_frames):
        frame = video[i].astype(np.float64)

        # Compute shift
        shift_x, shift_y, conf = compute_shift(reference, frame)

        # Clip to max shift
        shift_x = np.clip(shift_x, -max_shift, max_shift)
        shift_y = np.clip(shift_y, -max_shift, max_shift)

        shifts_x.append(float(shift_x))
        shifts_y.append(float(shift_y))
        confidences.append(float(conf))

        # Apply correction
        corrected = apply_shift(frame, shift_x, shift_y)
        corrected_video[i] = corrected.astype(video.dtype)

        if verbose and (i + 1) % 200 == 0:
            print(f"  Frame {i+1}/{n_frames}, last shift: ({shift_x:.2f}, {shift_y:.2f})")

    # Compute statistics
    shifts_x = np.array(shifts_x)
    shifts_y = np.array(shifts_y)
    magnitude = np.sqrt(shifts_x**2 + shifts_y**2)

    if verbose:
        print(f"\nDone!")
        print(f"Max shift: {magnitude.max():.2f} pixels")
        print(f"Mean shift: {magnitude.mean():.2f} pixels")
        print(f"Std shift: {magnitude.std():.2f} pixels")

    # Save results
    video_name = Path(video_path).stem

    if output_path is None:
        output_path = f"{video_name}_corrected.tif"

    if verbose:
        print(f"\nSaving corrected video to: {output_path}")
    tifffile.imwrite(output_path, corrected_video)

    # Save shifts
    shifts_path = Path(output_path).with_suffix('.shifts.json')
    shifts_data = {
        'shifts_x': shifts_x.tolist(),
        'shifts_y': shifts_y.tolist(),
        'confidences': confidences,
        'max_shift': float(magnitude.max()),
        'mean_shift': float(magnitude.mean()),
        'method': method,
        'reference_method': reference_method
    }
    with open(shifts_path, 'w') as f:
        json.dump(shifts_data, f, indent=2)
    if verbose:
        print(f"Saved shifts to: {shifts_path}")

    return {
        'corrected_video': corrected_video,
        'shifts_x': shifts_x,
        'shifts_y': shifts_y,
        'confidences': np.array(confidences),
        'reference': reference,
        'original_video': video
    }


def plot_motion_diagnostics(result: dict, save_path: str = None):
    """Plot motion correction diagnostics"""
    shifts_x = result['shifts_x']
    shifts_y = result['shifts_y']
    original = result['original_video']
    corrected = result['corrected_video']

    n_frames = len(shifts_x)
    magnitude = np.sqrt(shifts_x**2 + shifts_y**2)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Shifts over time
    axes[0, 0].plot(shifts_x, 'b-', linewidth=0.5, label='X shift')
    axes[0, 0].plot(shifts_y, 'r-', linewidth=0.5, label='Y shift')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Shift (pixels)')
    axes[0, 0].set_title('Detected Motion')
    axes[0, 0].legend()

    # Magnitude
    axes[0, 1].plot(magnitude, 'k-', linewidth=0.5)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Shift magnitude (pixels)')
    axes[0, 1].set_title(f'Shift Magnitude (max: {magnitude.max():.2f} px)')

    # XY scatter
    colors = np.arange(len(shifts_x))
    axes[0, 2].scatter(shifts_x, shifts_y, c=colors, cmap='viridis', s=2)
    axes[0, 2].set_xlabel('X shift (pixels)')
    axes[0, 2].set_ylabel('Y shift (pixels)')
    axes[0, 2].set_title('XY Motion Path')
    axes[0, 2].axis('equal')

    # Mean projection before
    axes[1, 0].imshow(np.mean(original, axis=0), cmap='gray')
    axes[1, 0].set_title('Mean Projection - Before')
    axes[1, 0].axis('off')

    # Mean projection after
    axes[1, 1].imshow(np.mean(corrected, axis=0), cmap='gray')
    axes[1, 1].set_title('Mean Projection - After')
    axes[1, 1].axis('off')

    # Difference
    diff = np.mean(corrected, axis=0).astype(float) - np.mean(original, axis=0).astype(float)
    im = axes[1, 2].imshow(diff, cmap='RdBu_r', vmin=-diff.std()*3, vmax=diff.std()*3)
    axes[1, 2].set_title('Difference (After - Before)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])

    plt.suptitle('Motion Correction Diagnostics', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostics to: {save_path}")

    plt.show()


def plot_frame_comparison(result: dict, frame_idx: int = None, save_path: str = None):
    """Compare before/after for a specific frame"""
    shifts_x = result['shifts_x']
    shifts_y = result['shifts_y']
    original = result['original_video']
    corrected = result['corrected_video']

    # Pick frame with largest shift if not specified
    if frame_idx is None:
        magnitude = np.sqrt(shifts_x**2 + shifts_y**2)
        frame_idx = np.argmax(magnitude)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original[frame_idx], cmap='gray')
    axes[0].set_title(f'Frame {frame_idx} - Before')
    axes[0].axis('off')

    axes[1].imshow(corrected[frame_idx], cmap='gray')
    axes[1].set_title(f'Frame {frame_idx} - After')
    axes[1].axis('off')

    diff = corrected[frame_idx].astype(float) - original[frame_idx].astype(float)
    axes[2].imshow(diff, cmap='RdBu_r')
    axes[2].set_title(f'Difference (shift: {shifts_x[frame_idx]:.1f}, {shifts_y[frame_idx]:.1f})')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python motion_correction_v2.py <video.tif>")
        print("  python motion_correction_v2.py <video.tif> --method template")
        print("  python motion_correction_v2.py <video.tif> --method xcorr")
        print("  python motion_correction_v2.py <video.tif> --reference first")
        print("")
        print("Methods: template (default, most robust), xcorr, phase")
        print("Reference: first (default), mean, middle")
        return

    video_path = sys.argv[1]

    # Parse arguments
    method = 'template'
    reference = 'first'

    if '--method' in sys.argv:
        idx = sys.argv.index('--method')
        method = sys.argv[idx + 1]

    if '--reference' in sys.argv:
        idx = sys.argv.index('--reference')
        reference = sys.argv[idx + 1]

    # Run correction
    result = motion_correct_video(
        video_path,
        method=method,
        reference_method=reference
    )

    # Plot diagnostics
    plot_motion_diagnostics(result, save_path='motion_diagnostics.png')
    plot_frame_comparison(result, save_path='motion_comparison.png')


if __name__ == "__main__":
    main()
