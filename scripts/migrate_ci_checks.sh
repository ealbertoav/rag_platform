#!/usr/bin/env bash
# Print current GitHub required checks and a migration template for T-174 CI job rename.
#
# Old checks (5 jobs): Dependency Scan, Lint, Unit Tests, Integration Tests, Retrieval Eval Regression
# New checks (3 jobs): Quality, Unit Tests, Extended Tests
#
# Usage:
#   ./scripts/migrate_ci_checks.sh
#   REPO=owner/name ./scripts/migrate_ci_checks.sh
set -euo pipefail

REPO="${REPO:-ealbertoav/rag_platform}"

echo "=== Repository: ${REPO} ==="
echo

echo "--- Branch protection (main) ---"
if gh api "repos/${REPO}/branches/main/protection" 2>/dev/null | jq -e '.required_status_checks.contexts' >/dev/null 2>&1; then
  gh api "repos/${REPO}/branches/main/protection" \
    --jq '.required_status_checks.contexts[]' 2>/dev/null || true
else
  echo "(no classic branch protection or not accessible)"
fi
echo

echo "--- Rulesets ---"
gh api "repos/${REPO}/rulesets" --jq '.[] | {id, name, enforcement, target}' 2>/dev/null || echo "(none or not accessible)"
echo

for ruleset_id in $(gh api "repos/${REPO}/rulesets" --jq '.[].id' 2>/dev/null || true); do
  echo "--- Ruleset ${ruleset_id} required checks ---"
  gh api "repos/${REPO}/rulesets/${ruleset_id}" \
    --jq '.rules[] | select(.type=="required_status_checks") | .parameters.required_status_checks[]?.context' \
    2>/dev/null || true
  echo
done

cat <<'EOF'
=== Migration template ===

Replace required status check contexts with:
  - Quality
  - Unit Tests
  - Extended Tests

Remove deprecated contexts:
  - Dependency Scan
  - Lint
  - Integration Tests
  - Retrieval Eval Regression

Example (after fetching RULESET_ID from above):

  gh api "repos/${REPO}/rulesets/{RULESET_ID}" > /tmp/ruleset.json
  # Edit required_status_checks contexts in /tmp/ruleset.json, then:
  gh api --method PUT "repos/${REPO}/rulesets/{RULESET_ID}" --input /tmp/ruleset.json

For classic branch protection:

  gh api --method PATCH "repos/${REPO}/branches/main/protection/required_status_checks" \
    -f strict=true \
    -f 'contexts[]=Quality' \
    -f 'contexts[]=Unit Tests' \
    -f 'contexts[]=Extended Tests'

Run after the first green CI on a PR that includes the T-174 workflow changes.
EOF
