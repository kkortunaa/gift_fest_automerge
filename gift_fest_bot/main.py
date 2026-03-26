from __future__ import annotations

import argparse
import json

from gift_fest_bot.bot import BotRuntime
from gift_fest_bot.config import BotConfig
from gift_fest_bot.controller.device import AdbDevice
from gift_fest_bot.controller.desktop import DesktopDevice
from gift_fest_bot.vision.template_matcher import TemplateRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gift Fest 2 autonomous screen bot")
    parser.add_argument(
        "mode",
        choices=["once", "loop", "doctor"],
        help="Run a single analysis pass, continuous loop, or ADB diagnostics",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--debug-dir", default=None, help="Directory for annotated output")
    parser.add_argument("--no-execute", action="store_true", help="Analyze only without sending input actions")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BotConfig.load(args.config)

    if args.mode == "doctor":
        recognizer = TemplateRecognizer(config.templates_dir, config.recognition)
        template_report = recognizer.template_stats()
        if config.control_mode == "adb":
            doctor_report = AdbDevice(config=config).doctor()
        elif config.control_mode == "desktop":
            doctor_report = DesktopDevice(config=config).doctor()
        else:
            raise SystemExit(f"doctor mode does not support control_mode={config.control_mode}")
        doctor_report["template_report"] = template_report
        print(json.dumps(doctor_report, indent=2))
        return

    runtime = BotRuntime.create(config)

    if args.mode == "once":
        summary = runtime.run_once(debug_dir=args.debug_dir, execute=not args.no_execute)
        print(json.dumps(summary, indent=2))
        return

    if args.no_execute:
        raise SystemExit("--no-execute is only supported with mode=once")

    runtime.run_loop(debug_dir=args.debug_dir)


if __name__ == "__main__":
    main()
