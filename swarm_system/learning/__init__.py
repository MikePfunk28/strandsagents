"""Learning modules for GAN-style feedback loops."""

from swarm_system.learning.generator_assistant import CodeCommentGeneratorAssistant
from swarm_system.learning.discriminator_assistant import CodeCommentDiscriminatorAssistant
from swarm_system.learning.agitator_assistant import CodeCommentAgitatorAssistant
from swarm_system.learning.code_feedback_loop import CodeFeedbackLoop
from swarm_system.learning.adaptive_benchmark import AdaptiveFeedbackBenchmark
from swarm_system.learning.adaptive_challenge_manager import AdaptiveChallengeManager

__all__ = [
    "CodeCommentGeneratorAssistant",
    "CodeCommentDiscriminatorAssistant",
    "CodeCommentAgitatorAssistant",
    "CodeFeedbackLoop",
    "AdaptiveFeedbackBenchmark",
    "AdaptiveChallengeManager",
]
