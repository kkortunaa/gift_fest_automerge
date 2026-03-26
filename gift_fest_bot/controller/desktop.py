from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import ImageGrab

from gift_fest_bot.config import BotConfig, Point

try:
    import dxcam
except ImportError:
    dxcam = None


INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
SW_RESTORE = 9


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class POINT_STRUCT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", INPUT_UNION)]


ctypes.windll.user32.SetProcessDPIAware()


@dataclass
class DesktopDevice:
    config: BotConfig
    _dxcam_camera: object | None = field(default=None, init=False, repr=False)
    _capture_region: tuple[int, int, int, int] = field(default=(0, 0, 0, 0), init=False, repr=False)
    _window_region_cache: tuple[int, int, int, int] | None = field(default=None, init=False, repr=False)
    _mini_app_region_cache: tuple[int, int, int, int] | None = field(default=None, init=False, repr=False)

    def capture_screen(self) -> np.ndarray:
        region = self._resolve_capture_region()
        frame = self._grab_with_backend(region)
        if frame is None:
            raise RuntimeError("Desktop capture failed")
        self._capture_region = region
        return frame

    def tap(self, point: Point) -> None:
        if self.config.desktop.focus_window:
            self._focus_target_window()
        absolute = self._to_absolute_point(point)
        self._set_cursor(absolute.x, absolute.y)
        time.sleep(0.05)
        self._send_mouse_flag(MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.03)
        self._send_mouse_flag(MOUSEEVENTF_LEFTUP)
        time.sleep(self.config.tap_post_delay_ms / 1000.0)

    def swipe(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if self.config.desktop.focus_window:
            self._focus_target_window()
        absolute_start = self._to_absolute_point(Point(*start))
        absolute_end = self._to_absolute_point(Point(*end))
        self._set_cursor(absolute_start.x, absolute_start.y)
        time.sleep(0.05)
        self._send_mouse_flag(MOUSEEVENTF_LEFTDOWN)

        steps = max(12, self.config.drag_duration_ms // 12)
        for index in range(1, steps + 1):
            x = int(absolute_start.x + (absolute_end.x - absolute_start.x) * index / steps)
            y = int(absolute_start.y + (absolute_end.y - absolute_start.y) * index / steps)
            self._set_cursor(x, y)
            time.sleep(self.config.drag_duration_ms / 1000.0 / steps)

        time.sleep(0.03)
        self._send_mouse_flag(MOUSEEVENTF_LEFTUP)
        time.sleep(self.config.swipe_post_delay_ms / 1000.0)

    def doctor(self) -> dict:
        hwnd = self._find_target_window()
        title = self._get_window_title(hwnd) if hwnd else None
        visible_titles = self._list_visible_window_titles(limit=20)
        window_region = self._resolve_window_region()
        region = self._resolve_capture_region()
        frame = self.capture_screen()
        return {
            "window_title_query": self.config.desktop.window_title,
            "window_found": bool(hwnd),
            "window_title": title,
            "visible_window_titles": visible_titles,
            "capture_backend": self._active_backend_name(),
            "auto_crop_to_mini_app": self.config.desktop.auto_crop_to_mini_app,
            "window_region": {
                "left": window_region[0],
                "top": window_region[1],
                "right": window_region[2],
                "bottom": window_region[3],
                "width": window_region[2] - window_region[0],
                "height": window_region[3] - window_region[1],
            },
            "mini_app_region": {
                "left": region[0],
                "top": region[1],
                "right": region[2],
                "bottom": region[3],
                "width": region[2] - region[0],
                "height": region[3] - region[1],
            },
            "capture_region": {
                "left": region[0],
                "top": region[1],
                "right": region[2],
                "bottom": region[3],
                "width": region[2] - region[0],
                "height": region[3] - region[1],
            },
            "frame_shape": {
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "channels": int(frame.shape[2]) if len(frame.shape) > 2 else 1,
            },
        }

    def _to_absolute_point(self, point: Point) -> Point:
        left, top, _, _ = self._capture_region
        return Point(x=left + int(point.x), y=top + int(point.y))

    def _grab_with_backend(self, region: tuple[int, int, int, int]) -> np.ndarray | None:
        backend = self.config.desktop.capture_backend
        if backend == "dxcam":
            frame = self._grab_dxcam(region)
            if frame is not None:
                return frame
        elif backend != "pil":
            raise ValueError(f"Unsupported desktop capture backend: {backend}")
        return self._grab_pil(region)

    def _grab_dxcam(self, region: tuple[int, int, int, int]) -> np.ndarray | None:
        if dxcam is None:
            return None
        if self._dxcam_camera is None:
            self._dxcam_camera = dxcam.create(output_color="BGR")
        try:
            frame = self._dxcam_camera.grab(region=region)
        except ValueError:
            return None
        if frame is None:
            return None
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return frame

    def _grab_pil(self, region: tuple[int, int, int, int]) -> np.ndarray:
        image = ImageGrab.grab(bbox=region, all_screens=True)
        rgb = np.array(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _resolve_capture_region(self) -> tuple[int, int, int, int]:
        window_region = self._resolve_window_region()
        if not self.config.desktop.auto_crop_to_mini_app:
            return window_region

        mini_app_region = self._detect_or_reuse_mini_app_region(window_region)
        if mini_app_region is not None:
            return mini_app_region
        return window_region

    def _resolve_window_region(self) -> tuple[int, int, int, int]:
        hwnd = self._find_target_window()
        if hwnd:
            if self.config.desktop.focus_window:
                self._focus_target_window(hwnd)
            region = self._window_region(hwnd)
            if region is not None:
                self._window_region_cache = region
                return region

        image = ImageGrab.grab(all_screens=True)
        width, height = image.size
        fallback = (0, 0, width, height)
        self._window_region_cache = fallback
        return fallback

    def _detect_or_reuse_mini_app_region(
        self,
        window_region: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        if self._mini_app_region_cache is not None and self._region_within(self._mini_app_region_cache, window_region):
            return self._mini_app_region_cache

        window_frame = self._grab_with_backend(window_region)
        if window_frame is None:
            return None

        local_region = self._detect_mini_app_region(window_frame)
        if local_region is None:
            return None

        absolute_region = (
            window_region[0] + local_region[0],
            window_region[1] + local_region[1],
            window_region[0] + local_region[2],
            window_region[1] + local_region[3],
        )
        self._mini_app_region_cache = absolute_region
        return absolute_region

    def _detect_mini_app_region(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        image_area = width * height

        bright_mask = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
        bright_mask = cv2.GaussianBlur(bright_mask, (7, 7), 0)
        _, bright_mask = cv2.threshold(bright_mask, 120, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = -1.0
        best_rect: tuple[int, int, int, int] | None = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < image_area * 0.12 or area > image_area * 0.95:
                continue
            if w < width * 0.25 or h < height * 0.35:
                continue
            if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
                continue

            ratio = w / max(h, 1)
            if not 0.35 <= ratio <= 1.15:
                continue

            roi = gray[y:y + h, x:x + w]
            brightness = float(np.mean(roi)) / 255.0
            centeredness = 1.0 - abs((x + w / 2) - width / 2) / max(width / 2, 1)
            vertical_bias = 1.0 - abs((y + h / 2) - height / 2) / max(height / 2, 1)
            score = brightness * 2.0 + area / image_area + centeredness + vertical_bias
            if score > best_score:
                best_score = score
                best_rect = (x, y, x + w, y + h)

        return best_rect

    def _region_within(
        self,
        inner: tuple[int, int, int, int],
        outer: tuple[int, int, int, int],
    ) -> bool:
        return (
            inner[0] >= outer[0]
            and inner[1] >= outer[1]
            and inner[2] <= outer[2]
            and inner[3] <= outer[3]
        )

    def _window_region(self, hwnd: int) -> tuple[int, int, int, int] | None:
        if self.config.desktop.client_only:
            client_rect = RECT()
            if not ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
                return None

            top_left = POINT_STRUCT(client_rect.left, client_rect.top)
            bottom_right = POINT_STRUCT(client_rect.right, client_rect.bottom)
            if not ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
                return None
            if not ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
                return None
            return top_left.x, top_left.y, bottom_right.x, bottom_right.y

        rect = RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return rect.left, rect.top, rect.right, rect.bottom

    def _focus_target_window(self, hwnd: int | None = None) -> None:
        target = hwnd or self._find_target_window()
        if not target:
            return
        ctypes.windll.user32.ShowWindow(target, SW_RESTORE)
        ctypes.windll.user32.SetForegroundWindow(target)
        time.sleep(0.08)

    def _find_target_window(self) -> int | None:
        query = self.config.desktop.window_title.strip().lower()
        if not query:
            return None

        matches: list[int] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def enum_windows(hwnd, _lparam):
            if not ctypes.windll.user32.IsWindowVisible(hwnd):
                return True
            title = self._get_window_title(hwnd)
            if title and query in title.lower():
                matches.append(int(hwnd))
            return True

        ctypes.windll.user32.EnumWindows(enum_windows, 0)
        return matches[0] if matches else None

    def _list_visible_window_titles(self, limit: int = 20) -> list[str]:
        titles: list[str] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def enum_windows(hwnd, _lparam):
            if not ctypes.windll.user32.IsWindowVisible(hwnd):
                return True
            title = self._get_window_title(hwnd).strip()
            if not title:
                return True
            titles.append(title)
            return len(titles) < limit

        ctypes.windll.user32.EnumWindows(enum_windows, 0)
        return titles

    def _get_window_title(self, hwnd: int) -> str:
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value

    def _active_backend_name(self) -> str:
        if self.config.desktop.capture_backend == "dxcam" and dxcam is not None:
            return "dxcam"
        if self.config.desktop.capture_backend == "dxcam":
            return "pil-fallback"
        return self.config.desktop.capture_backend

    def _set_cursor(self, x: int, y: int) -> None:
        ctypes.windll.user32.SetCursorPos(int(x), int(y))

    def _send_mouse_flag(self, flag: int) -> None:
        extra = ctypes.c_ulong(0)
        mouse_input = MOUSEINPUT(0, 0, 0, flag, 0, ctypes.pointer(extra))
        command = INPUT(INPUT_MOUSE, INPUT_UNION(mi=mouse_input))
        ctypes.windll.user32.SendInput(1, ctypes.byref(command), ctypes.sizeof(INPUT))
