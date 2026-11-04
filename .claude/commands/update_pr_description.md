---
description: Review all changes in the current branch's PR and rewrite its title and description
allowed-tools: Bash(git *), Bash(gh *), Bash(python3 *), Bash(curl *)
---

Update the title and description of the pull request for the current branch so they accurately reflect **all** of its changes.

## 1. Find the PR

- Current branch: `git branch --show-current`. If it is `main`, stop and tell the user there is no PR branch checked out.
- Remote repo: parse `owner/repo` from `git remote get-url origin`.
- Find the open PR whose head is this branch:
  - If `gh` is available: `gh pr view --json number,title,body,baseRefName,url`
  - Otherwise use the GitHub REST API with `$GITHUB_TOKEN` (or `$GH_TOKEN`):
    `GET https://api.github.com/repos/{owner}/{repo}/pulls?head={owner}:{branch}&state=open`
- If no open PR exists, stop and tell the user (do not create one).

## 2. Collect the changes

- `git fetch origin <base>` so the comparison is current.
- Warn the user if there are uncommitted changes or unpushed commits (`git status`, `git log origin/<branch>..HEAD`) — those are not part of the PR yet.
- Read every commit: `git log --reverse --format='%h %s%n%b' origin/<base>...origin/<branch>`
- Read the full diff: `git diff --stat origin/<base>...origin/<branch>` then `git diff origin/<base>...origin/<branch>`. For large generated files such as `uv.lock`, summarize only which notable packages changed version (`git diff ... -- uv.lock | grep -E '^[-+](name|version) ='`) instead of reading the whole diff.
- Read the existing PR body and keep any content still accurate that cannot be derived from the diff (e.g. linked issues, manual notes, checked test-plan items).

## 3. Write the title and description

Describe the PR's net effect against the base, not the commit history (if a later commit reverts or supersedes an earlier one, describe only the final state).

**Title**: concise (≤ 72 chars), imperative or release-style, covering the main purpose of the PR.

**Body** (GitHub Markdown):

```
## Summary
- <the main changes, grouped by theme: version/packaging, code, CI, dependencies, dev environment, ...>

## <Optional extra sections when relevant, e.g. "CI changes" or "Dependency upgrades" with old → new versions>

## Notes
- <breaking changes, dropped support, renamed/removed CI checks that affect branch protection, follow-ups>

## Test plan
- [ ] <how the change is verified; keep items already checked in the old body if still true>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

Omit empty sections. Be specific (file names, versions, Python/pandas matrices) but do not pad.

## 4. Update the PR

- Show the user the new title and body before updating.
- Update it:
  - With `gh`: `gh pr edit <number> --title "<title>" --body-file <file>` (write the body to a temp file in the scratchpad).
  - Otherwise: `PATCH https://api.github.com/repos/{owner}/{repo}/pulls/{number}` with JSON `{"title": ..., "body": ...}` and header `Authorization: Bearer $GITHUB_TOKEN`. Build the JSON with `python3` (`json.dumps`) to avoid quoting problems; never print the token.
- Report the PR URL when done.
