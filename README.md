# Gift Fest 2 Screen Bot

Autonomous Telegram mini-app bot for **Gift Fest 2** that works strictly from the screen:

- captures the Telegram Desktop or emulator window on Windows
- optionally captures an Android device through `adb`
- detects the 6x4 board automatically
- recognizes items from cropped cells
- plans the next action
- executes taps and drag-and-drop gestures

## Features

- Screen-only perception pipeline
- Window-targeted desktop capture with `dxcam` fallback to `Pillow`
- Automatic board detection with manual fallback
- Template matching baseline with ORB-assisted recognition
- Strategy engine that respects the main and secondary evolution chains
- Explicit state machine: scan -> plan -> execute -> wait_stable
- ADB input execution for tap, spawn, sell, and merge actions
- Windows desktop mouse control with client-area coordinate mapping
- Action-bar detection that distinguishes `sell`, `delete`, `create`, and `no_energy` from the visible UI
- Post-action stabilization check to avoid acting during merge animations

## Quick start

1. Install Python 3.11+.
2. Choose your control mode:
   - `desktop` for Telegram Desktop or BlueStacks on Windows
   - `adb` for Android device/emulator
3. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a template library:

```text
assets/templates/
  bear/
    001.png
  heart/
    001.png
  ...
```

4. Copy and edit the config:

```bash
copy config.example.yaml config.yaml
```

5. For `desktop`, set `desktop.window_title` in `config.yaml` to a substring of the target window title, usually `Telegram`.

6. Run diagnostics:

```bash
python -m gift_fest_bot.main doctor --config config.yaml
```

7. Run one analysis pass:

```bash
python -m gift_fest_bot.main once --config config.yaml --debug-dir debug
```

8. Run the autonomous loop:

```bash
python -m gift_fest_bot.main loop --config config.yaml --debug-dir debug
```

## Desktop workflow

Recommended path for Windows now:

1. Open Telegram Desktop and bring the mini app to the foreground.
2. Set `control_mode: desktop` in `config.yaml`.
3. Set `desktop.window_title` to match the target window.
4. Leave `desktop.auto_crop_to_mini_app: true` so the bot works only with the mini app area inside Telegram.
5. Leave Windows display scaling at 100% if possible.
6. Run `python -m gift_fest_bot.main doctor --config config.yaml`.
7. Confirm that `doctor` reports a valid `mini_app_region` and `frame_shape`.
8. Run `once --no-execute` and inspect `debug/last_frame.png`.
9. Only then run the autonomous loop.

Useful commands:

```bash
python -m gift_fest_bot.main doctor --config config.yaml
python -m gift_fest_bot.main once --config config.yaml --debug-dir debug --no-execute
python -m gift_fest_bot.main loop --config config.yaml --debug-dir debug
```

Desktop capture notes:

- `desktop.capture_backend: dxcam` is the fast path for dynamic mini apps.
- If `dxcam` is unavailable, the bot falls back to `Pillow`.
- `desktop.auto_crop_to_mini_app: true` tries to isolate the mini app viewport inside Telegram automatically.
- Input coordinates are mapped from the captured client area back to absolute screen coordinates before each click and drag.
- `desktop.focus_window: true` restores and foregrounds the target window before actions.

## ADB workflow

1. Enable VT-x / AMD-V in BIOS.
2. Install an Android emulator with ADB support or use a phone.
3. Launch Telegram and open the Gift Fest mini app.
4. Verify connection with `adb devices`.
5. Set `control_mode: adb` in `config.yaml`.
6. If multiple devices are visible, set `device_serial`.
7. Run `python -m gift_fest_bot.main doctor --config config.yaml`.
8. When `doctor` returns a valid screenshot size, run `once`, then `loop`.

## Calibration workflow

The bot can auto-detect the board, but a manual override is supported in `config.yaml` if the UI theme changes.

- Set `board_override` if auto-detection is unstable.
- Set `spawn_button`, `sell_button`, and `inventory_anchor` based on your screenshot if fallback coordinates are needed.
- Build templates from clean screenshots after each game art update.
- Add UI button templates under `assets/ui/sell/`, `assets/ui/delete/`, `assets/ui/create/`, and `assets/ui/no_energy/` for reliable action-bar classification.

## Strategy

Priority order:

1. Any available merge
2. Higher-level merges first
3. Main-chain-producing merges before pure secondary-chain merges
4. If no merges and empty cells exist: spawn
5. If the board is full: sell the lowest-value item

The value model explicitly protects items that are close to becoming main-chain items, such as `seeds`, `candles`, `vase`, and similar progression pieces.
The strategy also supports explicit delete policies from the real game, including "never delete", "last resort", and "prefer delete" buckets.

## Limits

- Recognition quality depends on good templates or a future trained detector.
- Lower-panel state detection works best when the action-bar templates are available.
- Board detection is heuristic unless manually overridden.
- Desktop mode depends on matching the correct window title and stable window geometry.
- Active-recognition clicks are more fragile than pure board matching and may still need to stay disabled in desktop mode.

## Next upgrade path

- Replace template recognition with a YOLO detector trained on cell crops.
- Plug in a detector through `gift_fest_bot/vision/detectors/yolo_adapter.py`.
- Add online confidence monitoring and auto-pausing on low-confidence frames.
- Add energy/animation-state detection to avoid acting during transitions.
