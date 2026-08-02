# Calibration profiles

Saved `CalibrationProfile` JSON files for `spectre/ingestion/image_digitizer.py`
-- each one records the pixel bounding box, curve color, and real axis range
for plots from one consistent source (e.g. one instrument's export
software), so you only run auto-detection once and reuse it for every future
image from that same source:

```python
from spectre.ingestion.image_digitizer import calibrate_and_save, digitize_plot

calibrate_and_save("sample_export.png", x_range=(4000, 400), y_range=(0, 1.05),
                    profile_path="calibration/my_instrument.json")

# later, on any new export from the same software:
spectrum = digitize_plot("new_export.png", profile_path="calibration/my_instrument.json")
```

`demo_instrument.json` here is the profile generated against
`examples/example_plot_image.png` during development/testing -- a working
example, not a real instrument calibration. Delete it and create your own
once you have real plot exports to calibrate against.
