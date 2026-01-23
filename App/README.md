# Bed Quality Checker - Mobile-Style Python App

A desktop application with a mobile phone interface for housekeepers to check bed quality using AI-powered image classification and defect detection.

## Features

- **Mobile-like Interface**: 375x812 window resembling an iPhone
- **Binary Classification**: Determines if bed is Made (0) or Unmade (1)
- **Defect Detection**: Identifies specific issues:
  - Items on bed (class 0)
  - Bed sheet not tucked (class 1)
  - Wrinkles (class 2)
- **Alignment Check**: Checks bed and pillow alignment
- **Supervisor Review**: Flags uncertain cases for manual review
- **Progress Tracking**: Tracks beds checked, approved, and issues raised

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

**Option A: Double-click**
- Double-click `run_app.bat`

**Option B: Command line**
```bash
python mobile_app.py
```

## How It Works

### Workflow

```
Select Image
    ↓
Binary Classifier (Stage1_Binary.pth)
    ↓
├─→ Good (High Confidence ≥0.85) → Alignment Check → Auto-approve or Review
├─→ Medium Confidence (0.6-0.85) → Send to Supervisor
└─→ Bad/Unmade or Low Conf (<0.6) → Defect Detection → Re-clean or Raise Issue
```

### Models Used

1. **Stage1_Binary.pth**: Binary classifier (Made vs Unmade)
2. **Stage2_Detection.pt**: YOLO model for defect detection
3. **alignment_scorer.py**: Alignment checking algorithm

## User Interface

### Camera Screen
- Stats display (Checked, Approved, Issues)
- Image selection button
- Status indicators

### Processing Screen
- Progress animation
- Image preview
- Status updates

### Result Screens
1. **Alignment Check**: For well-made beds
   - Shows alignment score
   - Options: Approve or Review again

2. **Defects Screen**: For unmade/defective beds
   - Lists all detected defects with confidence
   - Options: Re-clean or Raise issue

3. **Supervisor Review**: For uncertain cases
   - Shows classification confidence
   - Options: Send to supervisor or Re-clean

## File Structure

```
App/
├── mobile_app.py                    # Main application
├── alignment_scorer.py              # Alignment checking
├── api_server.py                    # API server (optional)
├── Stage1_Binary.pth               # Binary classifier model
├── Stage2_Detection.pt             # YOLO defect detector
├── Stage2_Detection.torchscript    # Alternative format
├── Stage3_BedPillow.pt             # Bed/pillow detector
├── requirements.txt                 # Python dependencies
├── run_app.bat                      # Windows launcher
└── Hotelroarz.drawio               # Flow diagram
```

## Keyboard Shortcuts

- **ESC**: Return to camera screen (from any screen)
- **Enter**: Trigger capture/action button

## Troubleshooting

### Models Not Loading

**Error**: "Failed to load models"

**Solution**:
- Ensure all model files are in the App directory
- Check if torch is properly installed: `pip install torch torchvision`
- Verify model files are not corrupted

### Image Selection Issues

**Error**: Can't select images

**Solution**:
- Make sure you have test bed images
- Supported formats: JPG, JPEG, PNG, BMP

### Alignment Check Fails

**Error**: Alignment scoring fails

**Solution**:
- Check if `alignment_scorer.py` is present
- Verify dependencies are installed
- The app will fallback to auto-approve if alignment check fails

### GUI Looks Wrong

**Error**: Interface doesn't look right

**Solution**:
```bash
pip install --upgrade customtkinter
```

## Configuration

### Confidence Thresholds

Edit in `mobile_app.py`:

```python
# Line ~200
if confidence >= 0.85:  # High confidence threshold
    self.check_alignment(confidence)
elif confidence >= 0.6:  # Medium confidence threshold
    self.show_supervisor_screen(...)
```

### Window Size

To change app dimensions (default: 375x812):

```python
# Line ~30
self.geometry("375x812")  # Change to desired size
```

## API Server (Optional)

If you want to run the models on a server instead:

1. Start the API server:
```bash
python api_server.py
```

2. Update `mobile_app.py` to use API calls instead of local models

## Development

### Adding New Features

1. **New Screen**: Create a new method like `show_XXX_screen()`
2. **New Action**: Add button with `command=self.your_action`
3. **Modify Flow**: Update `route_result()` method

### Customizing UI

Colors and styles use CustomTkinter theming:
- Primary color: `#2563eb` (blue)
- Success: `#10b981` (green)
- Warning: `#f59e0b` (orange)
- Danger: `#ef4444` (red)

## Performance Tips

- **First Run**: Model loading takes 5-10 seconds
- **Subsequent Runs**: Faster (models cached in memory)
- **Large Images**: Automatically resized for processing
- **Detection Speed**: YOLO inference ~0.5-2 seconds per image

## Testing

Test with various bed images:
1. **Perfect bed**: Should route to alignment → approve
2. **Messy bed**: Should detect defects
3. **Borderline bed**: Should route to supervisor

## Production Deployment

For deployment to housekeepers:

1. **Package as Executable**:
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --icon=icon.ico mobile_app.py
```

2. **Include Models**: Copy model files to dist folder

3. **Distribute**: Share the exe with model files

## Known Limitations

- Requires model files in same directory
- Desktop-only (not for actual mobile devices)
- No camera integration (file selection only)
- Offline operation only (unless using API server)

## Future Enhancements

- [ ] Live camera integration
- [ ] Cloud sync for results
- [ ] Multi-language support
- [ ] Export reports to PDF/Excel
- [ ] Room identification via QR codes
- [ ] Batch processing mode

## Credits

- UI Framework: CustomTkinter
- AI Models: PyTorch, Ultralytics YOLO
- Image Processing: OpenCV, Pillow

## Support

For issues or questions, check:
1. Model files are present and valid
2. All dependencies installed
3. Python 3.8+ is being used
4. No antivirus blocking the app

## License

Proprietary - All rights reserved
