use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use regex::Regex;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

/// Custom error types
#[derive(Error, Debug)]
pub enum ScreenshotError {
    #[error("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH")]
    FfmpegNotFound,

    #[error(
        "FFprobe not found. Please install FFmpeg (includes FFprobe) and ensure it's in your PATH"
    )]
    FfprobeNotFound,

    #[error("Input file not found: {0}")]
    InputNotFound(PathBuf),

    #[error("Invalid input file format: {0}")]
    InvalidFormat(String),

    #[error("Failed to get video duration: {0}")]
    DurationError(String),

    #[error("Invalid time format: {0}. Expected format: HH:MM:SS or seconds")]
    InvalidTimeFormat(String),

    #[error("Screenshot count must be greater than 0")]
    InvalidCount,

    #[error("Start time ({0}s) must be less than end time ({1}s)")]
    InvalidTimeRange(f64, f64),

    #[error("Time range exceeds video duration ({0}s)")]
    TimeExceedsDuration(f64),

    #[error("FFmpeg command failed: {0}")]
    FfmpegError(String),

    #[error("Failed to create output directory: {0}")]
    OutputDirError(String),
}

/// Output image format options
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum ImageFormat {
    #[default]
    Png,
    Jpg,
    Webp,
    Bmp,
}

impl ImageFormat {
    fn extension(&self) -> &str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::Jpg => "jpg",
            ImageFormat::Webp => "webp",
            ImageFormat::Bmp => "bmp",
        }
    }

    fn codec(&self) -> &str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::Jpg => "mjpeg",
            ImageFormat::Webp => "libwebp",
            ImageFormat::Bmp => "bmp",
        }
    }
}

/// Quality preset for output images
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum Quality {
    Low,
    #[default]
    Medium,
    High,
    Lossless,
}

impl Quality {
    fn get_quality_args(&self, format: &ImageFormat) -> Vec<String> {
        match format {
            ImageFormat::Jpg => {
                let q = match self {
                    Quality::Low => "15",
                    Quality::Medium => "5",
                    Quality::High => "2",
                    Quality::Lossless => "2",
                };
                vec!["-q:v".to_string(), q.to_string()]
            }
            ImageFormat::Webp => {
                let q = match self {
                    Quality::Low => "50",
                    Quality::Medium => "75",
                    Quality::High => "90",
                    Quality::Lossless => "100",
                };
                vec!["-quality".to_string(), q.to_string()]
            }
            ImageFormat::Png => {
                let compression = match self {
                    Quality::Low => "9",
                    Quality::Medium => "6",
                    Quality::High => "3",
                    Quality::Lossless => "0",
                };
                vec!["-compression_level".to_string(), compression.to_string()]
            }
            ImageFormat::Bmp => vec![],
        }
    }
}

/// CLI argument parser
#[derive(Parser, Debug)]
#[command(
    name = "vidshotter",
    author = "MLM Games",
    version = "1.0.3",
    about = "Extract equally-spaced screenshots from videos or GIFs",
    long_about = "A fast and parallel screenshot extractor that uses FFmpeg to capture equally-spaced frames from video files or GIFs."
)]
pub struct Args {
    /// Input video or GIF file path
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    /// Output directory for screenshots
    #[arg(short, long, value_name = "DIR")]
    output: Option<PathBuf>,

    /// Number of screenshots to extract
    #[arg(short = 'n', long, default_value = "30", value_name = "COUNT")]
    count: usize,

    /// Start time (format: HH:MM:SS, MM:SS, or seconds)
    #[arg(short = 's', long, value_name = "TIME")]
    start: Option<String>,

    /// End time (format: HH:MM:SS, MM:SS, or seconds)
    #[arg(short = 'e', long, value_name = "TIME")]
    end: Option<String>,

    /// Output image format
    #[arg(short = 'f', long, default_value = "png", value_enum)]
    format: ImageFormat,

    /// Image quality preset
    #[arg(short = 'q', long, default_value = "medium", value_enum)]
    quality: Quality,

    /// Output filename prefix
    #[arg(short = 'p', long, default_value = "screenshot")]
    prefix: String,

    /// Number of parallel extraction jobs
    #[arg(short = 'j', long, default_value = "4", value_name = "JOBS")]
    jobs: usize,

    /// Scale output images (e.g., "1920:-1" for 1920px width, maintaining aspect ratio)
    #[arg(long, value_name = "SCALE")]
    scale: Option<String>,

    /// Show verbose output
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Suppress progress bar and non-essential output
    #[arg(long)]
    quiet: bool,

    /// Show what would be done without executing
    #[arg(long)]
    dry_run: bool,

    /// Overwrite existing files without prompting
    #[arg(short = 'y', long)]
    overwrite: bool,
}

/// Video metadata from FFprobe
#[derive(Debug, Deserialize)]
struct FfprobeOutput {
    format: FfprobeFormat,
    streams: Vec<FfprobeStream>,
}

#[derive(Debug, Deserialize)]
struct FfprobeFormat {
    duration: Option<String>,
    filename: String,
    format_name: String,
}

#[derive(Debug, Deserialize)]
struct FfprobeStream {
    codec_type: String,
    width: Option<u32>,
    height: Option<u32>,
    duration: Option<String>,
    r_frame_rate: Option<String>,
    nb_frames: Option<String>,
}

/// Video information
#[derive(Debug)]
pub struct VideoInfo {
    pub duration: f64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub frame_rate: Option<f64>,
    pub format: String,
    pub filename: String,
}

/// Screenshot extraction job
#[derive(Debug, Clone)]
struct ExtractionJob {
    index: usize,
    timestamp: f64,
    output_path: PathBuf,
}

/// Main application struct
pub struct App {
    args: Args,
    ffmpeg_path: PathBuf,
    ffprobe_path: PathBuf,
}

impl App {
    /// Create a new App instance
    pub fn new(args: Args) -> Result<Self> {
        let ffmpeg_path = Self::find_executable("ffmpeg").ok_or(ScreenshotError::FfmpegNotFound)?;
        let ffprobe_path =
            Self::find_executable("ffprobe").ok_or(ScreenshotError::FfprobeNotFound)?;

        Ok(Self {
            args,
            ffmpeg_path,
            ffprobe_path,
        })
    }

    /// Find an executable in PATH
    fn find_executable(name: &str) -> Option<PathBuf> {
        // Check common locations first
        let common_paths = if cfg!(windows) {
            vec![
                format!("C:\\ffmpeg\\bin\\{}.exe", name),
                format!("C:\\Program Files\\ffmpeg\\bin\\{}.exe", name),
            ]
        } else {
            vec![
                format!("/usr/bin/{}", name),
                format!("/usr/local/bin/{}", name),
                format!("/opt/homebrew/bin/{}", name),
            ]
        };

        for path in common_paths {
            let p = PathBuf::from(&path);
            if p.exists() {
                return Some(p);
            }
        }

        // Try using 'which' or 'where' command
        let output = if cfg!(windows) {
            Command::new("where").arg(name).output()
        } else {
            Command::new("which").arg(name).output()
        };

        if let Ok(output) = output {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .next()
                    .map(|s| PathBuf::from(s.trim()));
                if let Some(p) = path {
                    if p.exists() {
                        return Some(p);
                    }
                }
            }
        }

        // Last resort: assume it's in PATH
        Some(PathBuf::from(name))
    }

    pub fn run(&self) -> Result<()> {
        self.validate_input()?;

        let video_info = self.get_video_info()?;

        if self.args.verbose {
            self.print_video_info(&video_info);
        }

        let (start_time, end_time) = self.calculate_time_range(&video_info)?;

        if self.args.count == 0 {
            return Err(ScreenshotError::InvalidCount.into());
        }

        let output_dir = self.create_output_dir()?;

        // Generate extraction jobs
        let jobs = self.generate_jobs(start_time, end_time, &output_dir)?;

        if self.args.dry_run {
            self.print_dry_run(&jobs, &video_info, start_time, end_time);
            return Ok(());
        }

        if !self.args.overwrite {
            self.check_existing_files(&jobs)?;
        }

        self.execute_extraction(&jobs)?;

        if !self.args.quiet {
            println!(
                "\n{} Successfully extracted {} screenshots to {}",
                "✓".green().bold(),
                jobs.len(),
                output_dir.display()
            );
        }

        Ok(())
    }

    /// Validate input file
    fn validate_input(&self) -> Result<()> {
        if !self.args.input.exists() {
            return Err(ScreenshotError::InputNotFound(self.args.input.clone()).into());
        }

        let extension = self
            .args
            .input
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        let valid_extensions = [
            "mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "m4v", "mpeg", "mpg", "3gp", "gif",
            "ts", "mts",
        ];

        if !valid_extensions.contains(&extension.as_str()) {
            return Err(ScreenshotError::InvalidFormat(extension).into());
        }

        Ok(())
    }

    /// Get video information using ffprobe
    fn get_video_info(&self) -> Result<VideoInfo> {
        let output = Command::new(&self.ffprobe_path)
            .args([
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
            ])
            .arg(&self.args.input)
            .output()
            .context("Failed to execute ffprobe")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ScreenshotError::DurationError(stderr.to_string()).into());
        }

        let probe: FfprobeOutput =
            serde_json::from_slice(&output.stdout).context("Failed to parse ffprobe output")?;

        // Get duration from format or first video stream
        let duration = probe
            .format
            .duration
            .as_ref()
            .and_then(|d| d.parse::<f64>().ok())
            .or_else(|| {
                probe
                    .streams
                    .iter()
                    .find(|s| s.codec_type == "video")
                    .and_then(|s| s.duration.as_ref())
                    .and_then(|d| d.parse::<f64>().ok())
            })
            .ok_or_else(|| {
                ScreenshotError::DurationError("Could not determine video duration".to_string())
            })?;

        // Get video stream info
        let video_stream = probe.streams.iter().find(|s| s.codec_type == "video");

        let (width, height) = video_stream
            .map(|s| (s.width, s.height))
            .unwrap_or((None, None));

        let frame_rate = video_stream
            .and_then(|s| s.r_frame_rate.as_ref())
            .and_then(|r| {
                let parts: Vec<&str> = r.split('/').collect();
                if parts.len() == 2 {
                    let num: f64 = parts[0].parse().ok()?;
                    let den: f64 = parts[1].parse().ok()?;
                    if den > 0.0 {
                        Some(num / den)
                    } else {
                        None
                    }
                } else {
                    r.parse().ok()
                }
            });

        Ok(VideoInfo {
            duration,
            width,
            height,
            frame_rate,
            format: probe.format.format_name,
            filename: probe.format.filename,
        })
    }

    /// Parse time string to seconds
    fn parse_time(time_str: &str) -> Result<f64> {
        let time_str = time_str.trim();

        // Try parsing as plain seconds first
        if let Ok(seconds) = time_str.parse::<f64>() {
            return Ok(seconds);
        }

        // Try HH:MM:SS or MM:SS format
        let time_regex = Regex::new(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")?;

        if let Some(captures) = time_regex.captures(time_str) {
            let hours: f64 = captures
                .get(1)
                .map(|m| m.as_str().parse().unwrap_or(0.0))
                .unwrap_or(0.0);
            let minutes: f64 = captures
                .get(2)
                .map(|m| m.as_str().parse().unwrap_or(0.0))
                .unwrap_or(0.0);
            let seconds: f64 = captures
                .get(3)
                .map(|m| m.as_str().parse().unwrap_or(0.0))
                .unwrap_or(0.0);

            return Ok(hours * 3600.0 + minutes * 60.0 + seconds);
        }

        Err(ScreenshotError::InvalidTimeFormat(time_str.to_string()).into())
    }

    /// Calculate time range for extraction
    fn calculate_time_range(&self, video_info: &VideoInfo) -> Result<(f64, f64)> {
        let start_time = match &self.args.start {
            Some(s) => Self::parse_time(s)?,
            None => 0.0,
        };

        let end_time = match &self.args.end {
            Some(e) => Self::parse_time(e)?,
            None => video_info.duration,
        };

        if start_time >= end_time {
            return Err(ScreenshotError::InvalidTimeRange(start_time, end_time).into());
        }

        if end_time > video_info.duration {
            return Err(ScreenshotError::TimeExceedsDuration(video_info.duration).into());
        }

        Ok((start_time, end_time))
    }

    /// Create output directory
    fn create_output_dir(&self) -> Result<PathBuf> {
        let output_dir = match &self.args.output {
            Some(dir) => dir.clone(),
            None => {
                let input_stem = self
                    .args
                    .input
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output");
                self.args
                    .input
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(format!("{}_screenshots", input_stem))
            }
        };

        fs::create_dir_all(&output_dir)
            .map_err(|e| ScreenshotError::OutputDirError(e.to_string()))?;

        Ok(output_dir)
    }

    /// Generate extraction jobs
    fn generate_jobs(
        &self,
        start_time: f64,
        end_time: f64,
        output_dir: &Path,
    ) -> Result<Vec<ExtractionJob>> {
        let duration = end_time - start_time;
        let count = self.args.count;

        // Calculate interval between screenshots
        let interval = if count == 1 {
            0.0
        } else {
            duration / (count as f64)
        };

        let jobs: Vec<ExtractionJob> = (0..count)
            .map(|i| {
                let timestamp = start_time + (i as f64 * interval) + (interval / 2.0);
                let timestamp = timestamp.min(end_time - 0.001);

                let filename = format!(
                    "{}_{:04}.{}",
                    self.args.prefix,
                    i + 1,
                    self.args.format.extension()
                );

                ExtractionJob {
                    index: i,
                    timestamp,
                    output_path: output_dir.join(filename),
                }
            })
            .collect();

        Ok(jobs)
    }

    fn check_existing_files(&self, jobs: &[ExtractionJob]) -> Result<()> {
        let existing: Vec<&PathBuf> = jobs
            .iter()
            .filter(|j| j.output_path.exists())
            .map(|j| &j.output_path)
            .collect();

        if !existing.is_empty() {
            eprintln!(
                "{} {} file(s) already exist. Use -y/--overwrite to replace them.",
                "Warning:".yellow().bold(),
                existing.len()
            );

            if self.args.verbose {
                for path in existing.iter().take(5) {
                    eprintln!("  - {}", path.display());
                }
                if existing.len() > 5 {
                    eprintln!("  ... and {} more", existing.len() - 5);
                }
            }

            return Err(anyhow!("Files already exist. Use -y to overwrite."));
        }

        Ok(())
    }

    /// Parallel extraction
    fn execute_extraction(&self, jobs: &[ExtractionJob]) -> Result<()> {
        let progress = if self.args.quiet {
            None
        } else {
            let pb = ProgressBar::new(jobs.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
                    .progress_chars("█▓░")
            );
            Some(pb)
        };

        let success_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));

        // Configure thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.args.jobs)
            .build()?;

        let input_path = &self.args.input;
        let format = &self.args.format;
        let quality = &self.args.quality;
        let scale = &self.args.scale;
        let ffmpeg_path = &self.ffmpeg_path;
        let overwrite = self.args.overwrite;
        let verbose = self.args.verbose;

        pool.install(|| {
            jobs.par_iter().for_each(|job| {
                let result = Self::extract_frame(
                    ffmpeg_path,
                    input_path,
                    job,
                    format,
                    quality,
                    scale,
                    overwrite,
                );

                match result {
                    Ok(_) => {
                        success_count.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(e) => {
                        error_count.fetch_add(1, Ordering::SeqCst);
                        if verbose {
                            eprintln!(
                                "{} Failed to extract frame {}: {}",
                                "Error:".red().bold(),
                                job.index + 1,
                                e
                            );
                        }
                    }
                }

                if let Some(ref pb) = progress {
                    pb.inc(1);
                }
            });
        });

        if let Some(pb) = progress {
            pb.finish_with_message("Done!");
        }

        let errors = error_count.load(Ordering::SeqCst);
        if errors > 0 {
            eprintln!(
                "\n{} {} screenshot(s) failed to extract",
                "Warning:".yellow().bold(),
                errors
            );
        }

        Ok(())
    }

    /// Single frame
    fn extract_frame(
        ffmpeg_path: &Path,
        input_path: &Path,
        job: &ExtractionJob,
        format: &ImageFormat,
        quality: &Quality,
        scale: &Option<String>,
        overwrite: bool,
    ) -> Result<()> {
        let mut cmd = Command::new(ffmpeg_path);

        // Overwrite flag
        if overwrite {
            cmd.arg("-y");
        } else {
            cmd.arg("-n");
        }

        // Seek to timestamp (before input for faster seeking)
        cmd.args(["-ss", &format!("{:.3}", job.timestamp)]);

        // Input file
        cmd.args(["-i"]).arg(input_path);

        // Extract single frame
        cmd.args(["-frames:v", "1"]);

        // Video codec
        cmd.args(["-c:v", format.codec()]);

        // Quality settings
        for arg in quality.get_quality_args(format) {
            cmd.arg(arg);
        }

        // Scale filter if specified
        if let Some(scale_value) = scale {
            cmd.args(["-vf", &format!("scale={}", scale_value)]);
        }

        // Output file
        cmd.arg(&job.output_path);

        // Suppress output
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::null());

        let status = cmd.status().context("Failed to execute ffmpeg")?;

        if !status.success() {
            return Err(ScreenshotError::FfmpegError(format!(
                "FFmpeg exited with status: {}",
                status
            ))
            .into());
        }

        Ok(())
    }

    fn print_video_info(&self, info: &VideoInfo) {
        println!("\n{}", "Video Information:".blue().bold());
        println!("  File: {}", info.filename);
        println!("  Format: {}", info.format);
        println!("  Duration: {}", Self::format_duration(info.duration));

        if let (Some(w), Some(h)) = (info.width, info.height) {
            println!("  Resolution: {}x{}", w, h);
        }

        if let Some(fps) = info.frame_rate {
            println!("  Frame Rate: {:.2} fps", fps);
        }
        println!();
    }

    fn print_dry_run(
        &self,
        jobs: &[ExtractionJob],
        video_info: &VideoInfo,
        start_time: f64,
        end_time: f64,
    ) {
        println!("\n{}", "Dry Run - Would extract:".yellow().bold());
        println!("  Input: {}", self.args.input.display());
        println!("  Duration: {}", Self::format_duration(video_info.duration));
        println!(
            "  Time Range: {} - {}",
            Self::format_duration(start_time),
            Self::format_duration(end_time)
        );
        println!("  Screenshots: {}", jobs.len());
        println!("  Format: {:?}", self.args.format);
        println!("  Quality: {:?}", self.args.quality);

        if let Some(ref scale) = self.args.scale {
            println!("  Scale: {}", scale);
        }

        println!("\n{}", "Timestamps:".blue().bold());
        for job in jobs.iter().take(10) {
            println!(
                "  {:4}: {} -> {}",
                job.index + 1,
                Self::format_duration(job.timestamp),
                job.output_path.display()
            );
        }

        if jobs.len() > 10 {
            println!("  ... and {} more", jobs.len() - 10);
        }
    }

    /// As HH:MM:SS.mmm
    fn format_duration(seconds: f64) -> String {
        let hours = (seconds / 3600.0) as u32;
        let minutes = ((seconds % 3600.0) / 60.0) as u32;
        let secs = seconds % 60.0;

        if hours > 0 {
            format!("{:02}:{:02}:{:06.3}", hours, minutes, secs)
        } else {
            format!("{:02}:{:06.3}", minutes, secs)
        }
    }
}

fn main() {
    let args = Args::parse();

    match App::new(args) {
        Ok(app) => {
            if let Err(e) = app.run() {
                eprintln!("{} {}", "Error:".red().bold(), e);

                // Print cause chain if present
                let mut cause = e.source();
                while let Some(c) = cause {
                    eprintln!("  Caused by: {}", c);
                    cause = c.source();
                }

                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("{} {}", "Error:".red().bold(), e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_time_seconds() {
        assert!((App::parse_time("10").unwrap() - 10.0).abs() < 0.001);
        assert!((App::parse_time("10.5").unwrap() - 10.5).abs() < 0.001);
        assert!((App::parse_time("0").unwrap() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_time_mmss() {
        assert!((App::parse_time("1:30").unwrap() - 90.0).abs() < 0.001);
        assert!((App::parse_time("01:30").unwrap() - 90.0).abs() < 0.001);
        assert!((App::parse_time("10:00").unwrap() - 600.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_time_hhmmss() {
        assert!((App::parse_time("1:00:00").unwrap() - 3600.0).abs() < 0.001);
        assert!((App::parse_time("01:30:00").unwrap() - 5400.0).abs() < 0.001);
        assert!((App::parse_time("2:30:45").unwrap() - 9045.0).abs() < 0.001);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(App::format_duration(90.0), "01:30.000");
        assert_eq!(App::format_duration(3661.5), "01:01:01.500");
        assert_eq!(App::format_duration(0.0), "00:00.000");
    }

    #[test]
    fn test_image_format_extension() {
        assert_eq!(ImageFormat::Png.extension(), "png");
        assert_eq!(ImageFormat::Jpg.extension(), "jpg");
        assert_eq!(ImageFormat::Webp.extension(), "webp");
        assert_eq!(ImageFormat::Bmp.extension(), "bmp");
    }

    #[test]
    fn test_invalid_time_format() {
        assert!(App::parse_time("invalid").is_err());
        assert!(App::parse_time("1:2:3:4").is_err());
    }
}
