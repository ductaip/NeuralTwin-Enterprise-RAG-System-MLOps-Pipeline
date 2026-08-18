"""Tool-call ceiling + loop detection, shared by both orchestrators.

Centralised because Bảng B (spec §3.3) is only a fair comparison if both orchestrators
enforce identical limits — a harness bug fixed in one place must fix it for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_TOOL_CALLS = 8


@dataclass
class ToolCallHarness:
    max_calls: int = MAX_TOOL_CALLS
    calls_used: int = 0
    _seen: dict[tuple, int] = field(default_factory=dict)

    @property
    def budget_exceeded(self) -> bool:
        return self.calls_used >= self.max_calls

    def record(self, tool_name: str, args: dict) -> str | None:
        """Record one call. Returns a warning observation string on the 2nd+ identical
        call (same tool, same args) — inserted as the tool result instead of failing,
        so the agent notices and changes course rather than looping silently."""
        self.calls_used += 1
        key = (tool_name, tuple(sorted(args.items())))
        self._seen[key] = self._seen.get(key, 0) + 1

        if self._seen[key] >= 2:
            return (
                f"You already called {tool_name}({args}) with these exact arguments — "
                f"repeating it will not surface new information. Use a different tool, "
                f"different arguments, or answer from the evidence already gathered."
            )
        return None
