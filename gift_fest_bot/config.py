from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class RecognitionConfig:
    match_threshold: float = 0.42
    unknown_threshold: float = 0.70
    empty_match_threshold: float = 0.9
    center_crop_ratio: float = 0.7
    strict_action_confidence_threshold: float = 0.9
    orb_good_match_threshold: int = 8
    cell_margin_ratio: float = 0.12
    board_left_ratio: float = 0.035
    board_right_ratio: float = 0.035
    board_top_ratio: float = 0.03
    board_bottom_ratio: float = 0.33
    ambiguity_margin: float = 0.05
    require_full_template_coverage: bool = False


@dataclass(frozen=True)
class ActionUiConfig:
    templates_dir: Path
    match_threshold: float = 0.72
    panel_height_ratio: float = 0.18
    top_margin_ratio: float = 0.02
    button_width_ratio: float = 0.36
    button_height_ratio: float = 0.5
    button_right_margin_ratio: float = 0.03
    button_vertical_margin_ratio: float = 0.2


@dataclass(frozen=True)
class DesktopCaptureConfig:
    window_title: str = "Telegram"
    focus_window: bool = True
    client_only: bool = True
    capture_backend: str = "dxcam"
    auto_crop_to_mini_app: bool = True


@dataclass(frozen=True)
class BotConfig:
    control_mode: str
    adb_path: str
    device_serial: Optional[str]
    desktop: DesktopCaptureConfig
    loop_delay_seconds: float
    debug: bool
    active_recognition_enabled: bool
    active_recognition_confidence_skip: float
    active_recognition_focus_click: bool
    debug_single_scan_cell: Optional[tuple[int, int]]
    active_recognition_debug: bool
    recognition: RecognitionConfig
    action_ui: ActionUiConfig
    templates_dir: Path
    board_override: Optional[tuple[int, int, int, int]]
    spawn_button: Point
    sell_button: Point
    inventory_anchor: Point
    drag_duration_ms: int
    tap_post_delay_ms: int
    swipe_post_delay_ms: int
    post_action_settle_ms: int
    stable_check_interval_ms: int
    stable_check_max_attempts: int
    stable_change_threshold: float

    @classmethod
    def load(cls, path: str | Path) -> "BotConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        board_override = data.get("board_override")
        board_tuple = tuple(board_override) if board_override else None

        return cls(
            control_mode=str(data.get("control_mode", "desktop")).lower(),
            adb_path=data.get("adb_path", "adb"),
            device_serial=data.get("device_serial"),
            desktop=DesktopCaptureConfig(
                window_title=str(data.get("desktop", {}).get("window_title", "Telegram")),
                focus_window=bool(data.get("desktop", {}).get("focus_window", True)),
                client_only=bool(data.get("desktop", {}).get("client_only", True)),
                capture_backend=str(data.get("desktop", {}).get("capture_backend", "dxcam")).lower(),
                auto_crop_to_mini_app=bool(data.get("desktop", {}).get("auto_crop_to_mini_app", True)),
            ),
            loop_delay_seconds=float(data.get("loop_delay_seconds", 1.0)),
            debug=bool(data.get("debug", True)),
            active_recognition_enabled=bool(data.get("active_recognition_enabled", True)),
            active_recognition_confidence_skip=float(data.get("active_recognition_confidence_skip", 0.98)),
            active_recognition_focus_click=bool(data.get("active_recognition_focus_click", True)),
            debug_single_scan_cell=tuple(data["debug_single_scan_cell"]) if data.get("debug_single_scan_cell") else None,
            active_recognition_debug=bool(data.get("active_recognition_debug", True)),
            recognition=RecognitionConfig(**data.get("recognition", {})),
            action_ui=ActionUiConfig(
                templates_dir=Path(data.get("action_ui", {}).get("templates_dir", "assets/ui")),
                match_threshold=float(data.get("action_ui", {}).get("match_threshold", 0.72)),
                panel_height_ratio=float(data.get("action_ui", {}).get("panel_height_ratio", 0.18)),
                top_margin_ratio=float(data.get("action_ui", {}).get("top_margin_ratio", 0.02)),
                button_width_ratio=float(data.get("action_ui", {}).get("button_width_ratio", 0.32)),
                button_height_ratio=float(data.get("action_ui", {}).get("button_height_ratio", 0.5)),
                button_right_margin_ratio=float(data.get("action_ui", {}).get("button_right_margin_ratio", 0.03)),
                button_vertical_margin_ratio=float(data.get("action_ui", {}).get("button_vertical_margin_ratio", 0.2)),
            ),
            templates_dir=Path(data.get("templates_dir", "assets/templates")),
            board_override=board_tuple,
            spawn_button=Point(**data["spawn_button"]),
            sell_button=Point(**data["sell_button"]),
            inventory_anchor=Point(**data["inventory_anchor"]),
            drag_duration_ms=int(data.get("drag_duration_ms", 180)),
            tap_post_delay_ms=int(data.get("tap_post_delay_ms", 250)),
            swipe_post_delay_ms=int(data.get("swipe_post_delay_ms", 350)),
            post_action_settle_ms=int(data.get("post_action_settle_ms", 250)),
            stable_check_interval_ms=int(data.get("stable_check_interval_ms", 220)),
            stable_check_max_attempts=int(data.get("stable_check_max_attempts", 8)),
            stable_change_threshold=float(data.get("stable_change_threshold", 0.02)),
        )
