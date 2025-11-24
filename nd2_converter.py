"""
ND2 to TIFF Converter for Voltage Imaging
Converts Nikon ND2 files to TIFF format for analysis
"""

import numpy as np
from pathlib import Path
import json


def convert_nd2_to_tiff(nd2_path: str, output_path: str = None,
                        chunk_size: int = 500, verbose: bool = True):
    """
    Convert ND2 file to TIFF

    Parameters:
    -----------
    nd2_path : str
        Path to input ND2 file
    output_path : str
        Path for output TIFF file (default: same name with .tif extension)
    chunk_size : int
        Number of frames to process at once (for memory management)
    verbose : bool
        Print progress

    Returns:
    --------
    dict : Metadata from the ND2 file
    """
    try:
        import nd2
    except ImportError:
        print("Please install nd2: uv pip install nd2")
        return None

    try:
        import tifffile
    except ImportError:
        print("Please install tifffile: uv pip install tifffile")
        return None

    nd2_path = Path(nd2_path)
    if output_path is None:
        output_path = nd2_path.with_suffix('.tif')
    else:
        output_path = Path(output_path)

    if verbose:
        print(f"Converting: {nd2_path}")
        print(f"Output: {output_path}")

    with nd2.ND2File(str(nd2_path)) as f:
        # Extract metadata
        if verbose:
            print(f"\n=== ND2 Metadata ===")
            print(f"Shape: {f.shape}")
            print(f"Dimensions: {f.sizes}")
            print(f"Data type: {f.dtype}")

        # Try to get frame rate
        fps = None
        try:
            # Method 1: from frame metadata
            if f.frame_metadata(0) is not None:
                fm = f.frame_metadata(0)
                if hasattr(fm, 'channels') and fm.channels:
                    if hasattr(fm.channels[0], 'time'):
                        # Get time difference between frames
                        t0 = f.frame_metadata(0).channels[0].time.relativeTimeMs
                        t1 = f.frame_metadata(1).channels[0].time.relativeTimeMs
                        frame_interval_ms = t1 - t0
                        fps = 1000 / frame_interval_ms
        except Exception:
            pass

        try:
            # Method 2: from experiment metadata
            if fps is None and hasattr(f, 'experiment'):
                for loop in f.experiment:
                    if hasattr(loop, 'parameters') and hasattr(loop.parameters, 'periodMs'):
                        fps = 1000 / loop.parameters.periodMs
                        break
        except Exception:
            pass

        try:
            # Method 3: from attributes
            if fps is None and hasattr(f, 'attributes'):
                if hasattr(f.attributes, 'widthPx'):
                    # Try to find timing info
                    pass
        except Exception:
            pass

        if fps:
            print(f"Frame rate: {fps:.2f} Hz")
        else:
            print("Frame rate: Could not determine (check metadata file)")

        # Get dimensions
        n_frames = f.sizes.get('T', f.shape[0] if len(f.shape) > 2 else 1)

        # Handle different dimension orders
        if verbose:
            print(f"\nTotal frames: {n_frames}")
            print(f"Converting...")

        # Load full array (if memory allows) or chunk-wise
        try:
            # Try loading full array
            video = f.asarray()

            # Squeeze out singleton dimensions
            video = np.squeeze(video)

            # Ensure 3D (T, Y, X)
            if video.ndim == 2:
                video = video[np.newaxis, :, :]
            elif video.ndim > 3:
                # Take first channel if multi-channel
                if verbose:
                    print(f"Multi-dimensional data {video.shape}, taking first channel")
                while video.ndim > 3:
                    video = video[:, 0] if video.shape[1] < video.shape[-1] else video[..., 0]

            if verbose:
                print(f"Final shape: {video.shape} (frames, height, width)")
                print(f"Data range: {video.min()} - {video.max()}")

            # Save as TIFF
            tifffile.imwrite(str(output_path), video, photometric='minisblack')

        except MemoryError:
            if verbose:
                print("File too large for memory, using chunked conversion...")

            # Chunked writing
            with tifffile.TiffWriter(str(output_path), bigtiff=True) as tif:
                for i in range(0, n_frames, chunk_size):
                    end = min(i + chunk_size, n_frames)
                    chunk = np.stack([f.read_frame(j) for j in range(i, end)])
                    chunk = np.squeeze(chunk)
                    tif.write(chunk, photometric='minisblack')
                    if verbose:
                        print(f"  Wrote frames {i}-{end} / {n_frames}")

        # Save metadata
        metadata = {
            'source_file': str(nd2_path),
            'output_file': str(output_path),
            'shape': list(video.shape) if 'video' in dir() else None,
            'sizes': dict(f.sizes),
            'dtype': str(f.dtype),
            'fps': fps,
            'n_frames': n_frames,
        }

        # Try to get more metadata
        try:
            if hasattr(f, 'attributes'):
                attrs = f.attributes
                if hasattr(attrs, 'widthPx'):
                    metadata['width_px'] = attrs.widthPx
                if hasattr(attrs, 'heightPx'):
                    metadata['height_px'] = attrs.heightPx
                if hasattr(attrs, 'pixelMicrons'):
                    metadata['pixel_microns'] = attrs.pixelMicrons
        except Exception:
            pass

        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as mf:
            json.dump(metadata, mf, indent=2)

        if verbose:
            print(f"\nDone!")
            print(f"TIFF saved to: {output_path}")
            print(f"Metadata saved to: {metadata_path}")
            if fps:
                print(f"\nUse fps={fps:.1f} when running analysis")

        return metadata


def batch_convert(input_dir: str, output_dir: str = None, pattern: str = "*.nd2"):
    """
    Convert all ND2 files in a directory

    Parameters:
    -----------
    input_dir : str
        Directory containing ND2 files
    output_dir : str
        Output directory (default: same as input)
    pattern : str
        Glob pattern for finding files
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    nd2_files = list(input_dir.glob(pattern))
    print(f"Found {len(nd2_files)} ND2 files")

    for i, nd2_file in enumerate(nd2_files):
        print(f"\n[{i+1}/{len(nd2_files)}] {nd2_file.name}")
        output_path = output_dir / nd2_file.with_suffix('.tif').name
        convert_nd2_to_tiff(nd2_file, output_path)

    print(f"\nBatch conversion complete!")


def inspect_nd2(nd2_path: str):
    """
    Print detailed metadata from ND2 file without converting
    """
    try:
        import nd2
    except ImportError:
        print("Please install nd2: uv pip install nd2")
        return

    print(f"Inspecting: {nd2_path}\n")

    with nd2.ND2File(str(nd2_path)) as f:
        print("=== Basic Info ===")
        print(f"Shape: {f.shape}")
        print(f"Sizes: {f.sizes}")
        print(f"Dtype: {f.dtype}")
        print(f"Ndim: {f.ndim}")

        print("\n=== Attributes ===")
        if hasattr(f, 'attributes') and f.attributes:
            attrs = f.attributes
            for attr in dir(attrs):
                if not attr.startswith('_'):
                    try:
                        val = getattr(attrs, attr)
                        if not callable(val):
                            print(f"  {attr}: {val}")
                    except Exception:
                        pass

        print("\n=== Metadata ===")
        if hasattr(f, 'metadata') and f.metadata:
            meta = f.metadata
            for attr in dir(meta):
                if not attr.startswith('_'):
                    try:
                        val = getattr(meta, attr)
                        if not callable(val):
                            print(f"  {attr}: {val}")
                    except Exception:
                        pass

        print("\n=== Frame 0 Metadata ===")
        try:
            fm = f.frame_metadata(0)
            if fm:
                print(f"  {fm}")
        except Exception as e:
            print(f"  Could not read: {e}")

        print("\n=== Experiment ===")
        try:
            if hasattr(f, 'experiment') and f.experiment:
                for i, loop in enumerate(f.experiment):
                    print(f"  Loop {i}: {loop}")
        except Exception as e:
            print(f"  Could not read: {e}")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python nd2_converter.py <file.nd2>              # Convert single file")
        print("  python nd2_converter.py <file.nd2> <output.tif> # Convert with custom output")
        print("  python nd2_converter.py --batch <directory>     # Convert all ND2 in directory")
        print("  python nd2_converter.py --inspect <file.nd2>    # Show metadata only")
        return

    if sys.argv[1] == '--batch':
        input_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        batch_convert(input_dir, output_dir)

    elif sys.argv[1] == '--inspect':
        if len(sys.argv) < 3:
            print("Please provide an ND2 file to inspect")
            return
        inspect_nd2(sys.argv[2])

    else:
        nd2_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        convert_nd2_to_tiff(nd2_path, output_path)


if __name__ == "__main__":
    main()
