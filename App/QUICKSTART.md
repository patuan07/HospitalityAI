# Quick Start Guide

## ✅ App Is Now Running!

A mobile-style window (375x812 pixels) should be open on your screen.

## How to Use

### 1. Wait for Models to Load
- The app will show "Loading AI models..." at the bottom
- Wait 5-10 seconds until it says "AI models loaded ✓"
- The "📷 Capture Photo" button will become active

### 2. Select a Bed Image
- Click the "📷 Capture Photo" button
- Browse and select a bed image (JPG, PNG, etc.)
- The app will automatically process it

### 3. Follow the Workflow

The app will automatically route based on the bed quality:

**Scenario A: Well-Made Bed (High Confidence)**
```
Select Image → Processing → Alignment Check → ✅ Approve
```

**Scenario B: Unmade/Defective Bed**
```
Select Image → Processing → Defects List → 🔄 Re-clean or ⚠️ Raise Issue
```

**Scenario C: Uncertain Quality**
```
Select Image → Processing → Supervisor Review → 📋 Send to Supervisor
```

### 4. Track Progress
- Top of screen shows statistics:
  - **Checked**: Total beds processed
  - **Approved**: Beds that passed
  - **Issues**: Beds with problems

## Keyboard Shortcuts

- **ESC**: Return to camera screen
- **Enter**: Activate current button

## Testing Tips

Test the app with different bed images:

1. **Perfect bed**: Clean, well-made → Should approve
2. **Messy bed**: Wrinkles, items → Should show defects
3. **Borderline bed**: → Should ask for supervisor

## Troubleshooting

### "Loading AI models..." stuck?
- Check terminal for errors
- Ensure model files exist:
  - `Stage1_Binary.pth`
  - `Stage2_Detection.pt`

### Can't select images?
- Make sure you have test bed images
- Supported: JPG, JPEG, PNG, BMP

### App crashed?
- Check terminal output
- Restart: `python mobile_app.py`

## Features

✅ Mobile-like interface (iPhone size)
✅ AI-powered classification
✅ Defect detection with YOLO
✅ Alignment checking
✅ Progress tracking
✅ Beautiful modern UI

## Next Steps

- **Run Again**: Double-click `run_app.bat`
- **API Mode**: Run `python api_server.py` for server mode
- **Customize**: Edit `mobile_app.py` to change thresholds

---

**Enjoy using the Bed Quality Checker!** 🛏️✨
