from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BotState(str, Enum):
    SCAN = "scan"
    PLAN = "plan"
    EXECUTE = "execute"
    WAIT_STABLE = "wait_stable"
    IDLE = "idle"


@dataclass
class StateMachine:
    state: BotState = BotState.SCAN

    def transition_after_scan(self) -> BotState:
        self.state = BotState.PLAN
        return self.state

    def transition_after_plan(self) -> BotState:
        self.state = BotState.EXECUTE
        return self.state

    def transition_after_execute(self) -> BotState:
        self.state = BotState.WAIT_STABLE
        return self.state

    def transition_after_stable(self) -> BotState:
        self.state = BotState.SCAN
        return self.state

    def to_idle(self) -> BotState:
        self.state = BotState.IDLE
        return self.state
