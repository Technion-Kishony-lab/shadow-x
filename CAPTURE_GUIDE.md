# Image Capture Scripts Guide

Two new scripts for capturing and saving images with your macroscope setup.

## 1. `capture_and_save.py` - Full Darkfield Session

**Purpose**: Captures complete brightfield + darkfield imaging session (like the demo images in README)

**What it does**:
- Captures 1 brightfield image (white backlight)
- Captures 50+ images with moving stripe patterns
- Processes into synthetic darkfield image
- Saves all results to `output/` directory

**Usage**:
```bash
conda activate macroscope
python capture_and_save.py
```

**Output files** (timestamped):
- `brightfield_YYYYMMDD_HHMMSS.jpg` - Simple brightfield image
- `darkfield_YYYYMMDD_HHMMSS_rgb.jpg` - Darkfield RGB image
- `darkfield_YYYYMMDD_HHMMSS_gray.jpg` - Darkfield grayscale
- `darkfield_data_YYYYMMDD_HHMMSS.pkl` - Raw capture data
- `reference_YYYYMMDD_HHMMSS_bright.jpg` - White reference
- `reference_YYYYMMDD_HHMMSS_dark.jpg` - Black reference

**Time**: ~30-60 seconds (depending on settings)

---

## 2. `quick_capture.py` - Single Image Capture

**Purpose**: Quickly capture single images with custom backlight colors

### Interactive Mode (default):
```bash
conda activate macroscope
python quick_capture.py
```

Commands:
- `w` - White backlight
- `b` - Black backlight  
- `r` - Red backlight
- `g` - Green backlight
- `rgb R G B` - Custom color (e.g., `rgb 128 64 255`)
- `q` - Quit

### Command Line Mode:
```bash
# White backlight
python quick_capture.py --white

# Black backlight
python quick_capture.py --black

# Custom RGB color
python quick_capture.py --color 255 128 0

# Custom output directory
python quick_capture.py --white --output my_images

# Adjust pause time
python quick_capture.py --white --pause 1.0
```

**Output**: `capture_YYYYMMDD_HHMMSS_[color].jpg` in `output/` directory

**Time**: < 2 seconds per image

---

## Comparison with Existing Scripts

| Script | Purpose | Saves Images? | Display |
|--------|---------|---------------|---------|
| `shadow.py` | Real-time shadow effect | ❌ No | ✅ Live on screens |
| `dark_field.py` | Darkfield demo | ❌ No | ✅ Matplotlib windows |
| **`capture_and_save.py`** | **Full darkfield capture** | **✅ Yes** | **✅ After capture** |
| **`quick_capture.py`** | **Quick single shots** | **✅ Yes** | **❌ No** |
| `test_camera_simple.py` | Camera test | ✅ Yes | ❌ No |

---

## Examples

### Recreate Demo Images
To create images like `demo_darkfield.jpeg` and `demo-brightfield.jpeg`:

```bash
python capture_and_save.py
```

Place your sample (e.g., plastic bag) on the macroscope before running.

### Quick Comparison Shots
```bash
python quick_capture.py --white    # Brightfield
python quick_capture.py --black    # Dark reference
python quick_capture.py --color 255 0 0  # Red backlight
```

### Custom Darkfield Parameters
Edit `capture_and_save.py` function `capture_full_session()`:

```python
darkfield_results = capture_darkfield_images(
    light_width=10,      # Narrower stripes
    dark_width=40,       # Wider dark regions
    num_images_to_discard=5,  # Fewer discards
    dark_dist=0.3,       # Tighter averaging
    smoothing=2,         # More smoothing
    output_dir=output_dir
)
```

---

## Tips

1. **Camera must be running**: Ensure digiCamControl is active
2. **Sample placement**: Place sample on the backlight screen before capture
3. **Focus**: Adjust camera focus in digiCamControl before running
4. **Processing time**: Darkfield processing takes ~5-10 seconds after capture
5. **Storage**: Each session creates ~500KB-2MB of images

---

## Troubleshooting

**Error: "No module named 'cv2'"**
- Activate conda environment: `conda activate macroscope`

**Error: "Failed to connect to digiCamControl"**
- Ensure digiCamControl is running
- Check web server is enabled on port 5514

**Images too dark/bright**
- Adjust camera exposure in digiCamControl
- Modify `dark_dist` parameter for darkfield

**Windows not displaying**
- Check screen configuration in `env_macroscope.py`
- Try running `test_setup.py` first

