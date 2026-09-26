# Source this (do not execute):   . scripts/env.sh
# Loads local credential files and points the agent at DigitalOcean serverless inference.
#
# Precedence: .env.digitalocean (infra token) -> .env (project settings, may hold OPENAI_API_KEY)
# -> existing shell values win for the three exports below.
#
# Keys:
#   * Preferred: a Gradient *model access key* in .env as OPENAI_API_KEY (inference-only, least privilege;
#     created in the control panel under Gradient AI Platform > Serverless inference > Model access keys).
#   * Stopgap: the DO personal access token (DIGITALOCEAN_ACCESS_TOKEN) also authenticates against
#     inference.do-ai.run, but it is a full infra credential - fall back to it only when no model key exists.
# Never echo these values; this file prints nothing.
set -a
[ -f .env.digitalocean ] && . ./.env.digitalocean
[ -f .env ] && . ./.env
set +a

export OPENAI_API_KEY="${OPENAI_API_KEY:-${DIGITALOCEAN_ACCESS_TOKEN:-}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://inference.do-ai.run/v1}"
export MODEL_ID="${MODEL_ID:-openai-gpt-4o-mini}"
