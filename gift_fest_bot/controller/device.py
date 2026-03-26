from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

import cv2
import numpy as np

from gift_fest_bot.config import BotConfig, Point


@dataclass
class AdbDevice:
    config: BotConfig

    def _base_command(self) -> list[str]:
        command = [self.config.adb_path]
        if self.config.device_serial:
            command.extend(["-s", self.config.device_serial])
        return command

    def _run(self, args: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            self._base_command() + args,
            check=True,
            capture_output=capture_output,
        )

    def doctor(self) -> dict:
        adb_version = subprocess.run(
            [self.config.adb_path, "version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        devices_output = subprocess.run(
            [self.config.adb_path, "devices"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        device_state = self._run(["get-state"], capture_output=True).stdout.decode("utf-8", errors="replace").strip()
        wm_size = self._run(["shell", "wm", "size"], capture_output=True).stdout.decode("utf-8", errors="replace").strip()
        display_density = self._run(["shell", "wm", "density"], capture_output=True).stdout.decode(
            "utf-8", errors="replace"
        ).strip()

        screenshot = self.capture_screen()
        screenshot_shape = {
            "width": int(screenshot.shape[1]),
            "height": int(screenshot.shape[0]),
            "channels": int(screenshot.shape[2]) if len(screenshot.shape) > 2 else 1,
        }

        return {
            "adb_path": self.config.adb_path,
            "device_serial": self.config.device_serial,
            "adb_version": adb_version,
            "devices_output": devices_output,
            "device_state": device_state,
            "wm_size": wm_size,
            "wm_density": display_density,
            "screencap": screenshot_shape,
        }

    def capture_screen(self) -> np.ndarray:
        result = self._run(["exec-out", "screencap", "-p"], capture_output=True)
        image_array = np.frombuffer(result.stdout, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("ADB screen capture failed")
        return image

    def tap(self, point: Point) -> None:
        self._run(["shell", "input", "tap", str(point.x), str(point.y)])
        time.sleep(self.config.tap_post_delay_ms / 1000.0)

    def swipe(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        self._run(
            [
                "shell",
                "input",
                "swipe",
                str(start[0]),
                str(start[1]),
                str(end[0]),
                str(end[1]),
                str(self.config.drag_duration_ms),
            ]
        )
        time.sleep(self.config.swipe_post_delay_ms / 1000.0)
