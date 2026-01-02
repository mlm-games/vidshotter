# Vidshotter

A fast, parallel CLI tool to extract equally-spaced screenshots from videos or GIFs using FFmpeg.

## Features

- 🚀 **Parallel extraction** - Configurable number of concurrent jobs
- 📊 **Progress bar** - Visual feedback during extraction
- 🎯 **Precise timing** - Specify start/end times in multiple formats
- 🖼️ **Multiple formats** - PNG, JPG, WebP, BMP output support
- ⚡ **Quality presets** - Low, Medium, High, Lossless options
- 📐 **Scaling** - Resize output images
- 🔍 **Dry run** - Preview what would be extracted

## Installation

### Prerequisites

- FFmpeg (install from https://ffmpeg.org or via package manager)

### Build from source

```bash
git clone https://github.com/mlm-games/vidshotter
cd vidshotter
cargo build --release
```

The binary will be at `target/release/vidshotter`

## Usage

### Basic usage (30 screenshots, auto timing)

```bash
vidshotter -i video.mp4
```

### Specify number of screenshots

```bash
vidshotter -i video.mp4 -n 50
```

### Custom output directory

```bash
vidshotter -i video.mp4 -o ./my_screenshots
```

### Specify time range

```bash
# Using seconds
vidshotter -i video.mp4 -s 10 -e 120

# Using MM:SS format
vidshotter -i video.mp4 -s 1:30 -e 5:00

# Using HH:MM:SS format
vidshotter -i video.mp4 -s 0:01:30 -e 0:10:00
```

### Change output format and quality

```bash
vidshotter -i video.mp4 -f jpg -q high
vidshotter -i video.mp4 -f webp -q medium
```

### Scale output images

```bash
# Width 1920px, maintain aspect ratio
vidshotter -i video.mp4 --scale 1920:-1

# Height 720px, maintain aspect ratio
vidshotter -i video.mp4 --scale -1:720

# Exact size
vidshotter -i video.mp4 --scale 1280:720
```

### Parallel jobs

```bash
# Use 8 parallel extraction jobs
vidshotter -i video.mp4 -j 8
```

### Dry run

```bash
vidshotter -i video.mp4 -n 20 --dry-run
```

### Full example

```bash
vidshotter \
  -i movie.mkv \
  -o ./frames \
  -n 100 \
  -s 5:00 \
  -e 1:30:00 \
  -f png \
  -q high \
  -p frame \
  -j 8 \
  --scale 1920:-1 \
  -v \
  -y
```

## Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input video/GIF file | Required |
| `--output` | `-o` | Output directory | `<input>_screenshots` |
| `--count` | `-n` | Number of screenshots | 30 |
| `--start` | `-s` | Start time | 0 |
| `--end` | `-e` | End time | Video duration |
| `--format` | `-f` | Image format (png/jpg/webp/bmp) | png |
| `--quality` | `-q` | Quality preset | medium |
| `--prefix` | `-p` | Filename prefix | screenshot |
| `--jobs` | `-j` | Parallel jobs | 4 |
| `--scale` | | Output scaling | None |
| `--verbose` | `-v` | Verbose output | false |
| `--quiet` | | Suppress output | false |
| `--dry-run` | | Preview only | false |
| `--overwrite` | `-y` | Overwrite existing | false |

## Supported Formats

### Input
MP4, MKV, AVI, MOV, WMV, FLV, WebM, M4V, MPEG, MPG, 3GP, GIF, TS, MTS

### Output
PNG, JPG, WebP, BMP

## License

GPL-3.0 License

## Building and Running

```bash
# Build in release mode
cargo build --release

# Run with a video file
./target/release/vidshotter -i video.mp4

# Install globally (optional)
cargo install --path .

# Then run from anywhere
vidshotter -i video.mp4 -n 50
```

## Example Commands

```bash
# Extract 30 screenshots from a video (default)
vidshotter -i movie.mp4

# Extract 50 screenshots as JPG with high quality
vidshotter -i movie.mp4 -n 50 -f jpg -q high

# Extract screenshots from 1 minute to 5 minutes
vidshotter -i movie.mp4 -s 1:00 -e 5:00 -n 20

# Extract with custom prefix and output directory
vidshotter -i movie.mp4 -o ./frames -p scene -n 100

# Preview extraction without doing it
vidshotter -i movie.mp4 -n 100 --dry-run -v

# Extract from GIF
vidshotter -i animation.gif -n 10 -f png
```