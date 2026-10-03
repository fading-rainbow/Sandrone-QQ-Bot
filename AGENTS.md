# Source synchronization and Cloud workflow

- Read `SOURCE_BASELINE.json` when deciding which branch or release to use. `main` is the reviewed current development source; it is not a claim that every development change has been deployed.
- Before editing, inspect the current branch, remote HEAD, and local changes. On an existing dirty Cloud checkout, preserve user changes; never reset or discard them to refresh a snapshot.
- After completing requested code changes, run appropriate checks, scan the new publication for secrets, and synchronize reviewed work to the correct GitHub development branch. Report exact commit and checks. No force push and no unreviewed bulk commit of unrelated work.
- Keep production release source/image/version distinct. Update production evidence only after an authorized deployment and live verification. Source synchronization never authorizes deployment/restart/database mutation.
- Never publish `.env`, credentials, Codex authentication, databases, chat histories, logs, raw captures, signing keys, backups, or runtime configuration. Public source removes real QQ identity defaults; configure identities through environment variables locally.
- Cloud is for isolated development and mocked tests. Do not start production bots or access VPS/real QQ/model accounts unless explicitly requested. Android/Windows runtime checks require their actual target platform; do not label Linux static checks as platform acceptance.
- Preserve upstream attribution/licenses, use established tools, and keep VPS work lightweight. New operational project repositories should be private by default.
