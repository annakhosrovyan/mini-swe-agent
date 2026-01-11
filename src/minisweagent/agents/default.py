"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import fnmatch
import re
import subprocess
import time

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model


class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0
    max_history_messages: int = 0
    validate_source_changes: bool = False
    source_path_patterns: list[str] = ["src/**/*.py", "*.py"]


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = self.config.model_dump() | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        msgs = self.messages
        if self.config.max_history_messages > 0 and len(msgs) > 2:
            head = msgs[:2]
            tail = msgs[2:][-self.config.max_history_messages :]
            msgs = head + tail
        response = self.model.query(msgs)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        self.has_finished(output)
        return output | {"action": action["action"]}

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            submission = "".join(lines[1:])
            if self.config.validate_source_changes:
                if not self._has_source_file_changes(submission):
                    raise FormatError(
                        "Submission rejected: No source files were modified. "
                        "You must modify at least one source file (matching patterns: "
                        + ", ".join(self.config.source_path_patterns)
                        + ") for the submission to be valid."
                    )
            raise Submitted(submission)
    
    def _has_source_file_changes(self, git_diff_output: str) -> bool:
        """Check if git diff output contains changes to source files."""
        changed_files = []
        for line in git_diff_output.splitlines():
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 3:
                    filepath = parts[2].lstrip("a/").lstrip("b/")
                    changed_files.append(filepath)
            elif line.startswith("+++") or line.startswith("---"):
                parts = line.split()
                if len(parts) >= 2:
                    filepath = parts[1].lstrip("a/").lstrip("b/")
                    if filepath != "/dev/null":
                        changed_files.append(filepath)
        
        for filepath in changed_files:
            for pattern in self.config.source_path_patterns:
                if fnmatch.fnmatch(filepath, pattern) or filepath.endswith(".py"):
                    return True
        return False