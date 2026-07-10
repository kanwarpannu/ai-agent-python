from __future__ import annotations

import logging

from .models import AppState, RAGContext, TrajectoryStep

logger = logging.getLogger(__name__)


class StateMachine:
    def __init__(self, context: RAGContext) -> None:
        self.context = context

    @property
    def state(self) -> AppState:
        return self.context.state

    def transition(self, new_state: AppState) -> None:
        logger.info("STATE %s -> %s", self.context.state.value, new_state.value)
        self.context.state = new_state
        self.context.trajectory.append(
            TrajectoryStep(step="transition", state=new_state)
        )

    def fail(self, exc: Exception) -> None:
        self.context.last_error = str(exc)
        self.transition(AppState.ERROR)
