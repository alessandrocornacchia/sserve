# Issue automation

Two GitHub Actions workflows let Claude plan and implement issues, with a
maintainer approving the plan in between.

```
issue opened ──► claude-triage.yml ──► plan comment, label plan-proposed
                                            │
                       maintainer reviews, comments, adds `ready`
                                            ▼
             claude-implement.yml ──► branch claude/issue-<n>, tested on a
                                      Docker SLURM cluster, draft PR,
                                      label draft-complete
```

## Labels

The triage workflow creates these on its first run.

| Label | Who adds it | Meaning |
|---|---|---|
| `needs-plan` | maintainer | (Re)plan this issue. Use it for issues opened by people without write access, which are not planned automatically, or to get a revised plan after commenting. |
| `plan-proposed` | Claude | A plan is posted and waits for review. |
| `ready` | maintainer | The plan is approved: implement it. |
| `draft-complete` | Claude | A draft PR is open for you to test. |

Triage also adds one of `bug`, `enhancement`, `documentation`, `question`.

## Reviewing a plan

The plan is a comment starting with "Implementation plan". To change it,
either comment on the issue (the implementation treats maintainers' comments
after the plan as amendments) or comment and add `needs-plan` to get a
rewritten plan. Then add `ready`.

Only comments from the owner, members and collaborators are passed to Claude,
so other people's comments can't change the plan.

## The pull request

Claude implements on `claude/issue-<n>` and the workflow opens a draft PR that
closes the issue. Re-adding `ready` redoes the implementation from the default
branch and force-pushes the same branch, updating the PR. If a run fails, the
workflow comments on the issue with a link to the logs and removes `ready`.

Claude can't reach a real cluster from CI. Instead the workflow starts the
Docker SLURM cluster from [testing](testing.md), where Claude tries its change
with a mock vLLM and extends `tests/slurm/smoke.sh`. The workflow then reruns the
smoke test and puts the result in the PR (a failing test doesn't stop the PR),
and the PR lists what to check on a real GPU cluster.

## Setup

1. Add a repository secret: `ANTHROPIC_API_KEY`, or `CLAUDE_CODE_OAUTH_TOKEN`
   (from `claude setup-token`, uses a Claude subscription).
2. Allow the workflows to open PRs: Settings → Actions → General → Workflow
   permissions → "Allow GitHub Actions to create and approve pull requests".

PRs opened by the workflow don't trigger other workflows (a GitHub rule for
`GITHUB_TOKEN`), so the SLURM test workflow doesn't run on them by itself;
start it from the Actions tab ("Run workflow" on the PR's branch) if you need it.
