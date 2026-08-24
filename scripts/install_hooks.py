# Copyright 2026 CoreField (Furqan Shakeel)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install a pre-commit hook that blocks confidential terms from being committed.

    python scripts/install_hooks.py

WHY THE TERM LIST IS NOT IN THIS REPOSITORY
--------------------------------------------
The obvious way to enforce "these words must never be committed" is a CI job
containing those words. That publishes them -- a signpost pointing at exactly
what is being protected, which is worse than no check at all.

So the list lives in `.confidential-terms`, which is gitignored and never
leaves the author's machine. This script installs a local pre-commit hook
that reads it and blocks any commit whose staged content matches. CI performs
only the structural check it can do without naming anything: that the
sensitive directories are not tracked.

The file is one pattern per line, `#` for comments. Case-insensitive,
extended-regex. It is created with an explanatory header on first run if it
does not exist.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TERMS_FILE = ".confidential-terms"

TERMS_TEMPLATE = """\
# Confidential terms -- one extended-regex pattern per line, '#' for comments.
#
# THIS FILE IS GITIGNORED AND MUST STAY THAT WAY. It exists so the pre-commit
# hook can block these terms without the terms themselves ever entering the
# repository.
#
# Add the pilot host's employer, job title, location, and any other
# identifying detail. Matching is case-insensitive.
#
# Examples of the KIND of thing that belongs here (replace with real values):
#   <employer-name>
#   <city-name>
#   <job-title>
"""

HOOK = r"""#!/bin/sh
# CoreField pre-commit confidentiality gate. Installed by scripts/install_hooks.py.
# Blocks any staged content matching a pattern in .confidential-terms.
set -eu

TERMS=".confidential-terms"
[ -f "$TERMS" ] || exit 0

PATTERNS=$(grep -vE '^\s*(#|$)' "$TERMS" 2>/dev/null | paste -sd'|' -) || PATTERNS=""
[ -n "$PATTERNS" ] || exit 0

HITS=$(git diff --cached --name-only --diff-filter=ACM | while IFS= read -r f; do
  [ -f "$f" ] || continue
  if git show ":$f" 2>/dev/null | grep -nIiE "$PATTERNS" >/dev/null 2>&1; then
    echo "  $f"
  fi
done)

if [ -n "$HITS" ]; then
  echo ""
  echo "COMMIT BLOCKED - confidential terms found in staged content:"
  echo "$HITS"
  echo ""
  echo "Per CLAUDE.md, the pilot host's employer, title and identifying detail"
  echo "must never appear in this repository, in commit messages, or in issues."
  echo "Remove the detail, or refer only to 'the pilot host'."
  echo ""
  echo "This hook is a safety net, not a decision-maker. If you believe a match"
  echo "is a false positive, review it deliberately before using --no-verify."
  echo ""
  exit 1
fi
exit 0
"""


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    git_dir = root / ".git"
    if not git_dir.exists():
        print(
            f"No .git directory at {root}. Run `git init` first -- and note that the "
            f"first commit is the one that matters most here, so install this hook "
            f"before making it.",
            file=sys.stderr,
        )
        return 1

    terms = root / TERMS_FILE
    if not terms.exists():
        terms.write_text(TERMS_TEMPLATE, encoding="utf-8")
        print(f"Created {TERMS_FILE} (gitignored). Add the terms to block, one per line.")
    else:
        print(f"{TERMS_FILE} already exists; leaving it alone.")

    hooks_dir = Path(
        subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
    )
    if not hooks_dir.is_absolute():
        hooks_dir = root / hooks_dir
    hooks_dir.mkdir(parents=True, exist_ok=True)

    hook_path = hooks_dir / "pre-commit"
    if hook_path.exists() and "CoreField pre-commit confidentiality gate" not in hook_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        backup = hook_path.with_suffix(".pre-corefield.bak")
        backup.write_bytes(hook_path.read_bytes())
        print(f"Existing pre-commit hook backed up to {backup.name}")

    hook_path.write_text(HOOK, encoding="utf-8", newline="\n")
    hook_path.chmod(0o755)
    print(f"Installed pre-commit hook at {hook_path}")

    patterns = [
        line
        for line in terms.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not patterns:
        print(
            "\nWARNING: .confidential-terms currently contains no patterns, so the hook "
            "will pass everything. Add the terms you need blocked before committing."
        )
    else:
        print(f"{len(patterns)} pattern(s) active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
