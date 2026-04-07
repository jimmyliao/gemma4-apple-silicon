#!/usr/bin/env bash
# m1_fix_json_files.sh
# Fix JSON files corrupted by gws CLI's stdout reformatting.
#
# Background: gws drive files get for application/json files dumps
# the parsed JSON content to stdout (re-serialized, not byte-identical).
# This breaks downstream tools that need exact JSON structure (e.g.,
# mlx-lm needing nested config fields like text_config.vocab_size).
#
# Solution: Use the OAuth refresh token from `gws auth export`, exchange
# for an access token, and call Drive REST API directly with curl.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DIR="$REPO_ROOT/models/gemma-4-e2b-it-mlx-4bit"
FOLDER_NAME="gemma-4-e2b-it-mlx-4bit"

gws_json() { gws "$@" 2>/dev/null | sed -n '/^{/,$p'; }

# 1. Get refresh token + client credentials
echo "→ Extracting OAuth credentials from gws keyring..."
CREDS=$(gws auth export --unmasked 2>/dev/null | sed -n '/^{/,$p')
CLIENT_ID=$(echo "$CREDS" | jq -r '.client_id')
CLIENT_SECRET=$(echo "$CREDS" | jq -r '.client_secret')
REFRESH_TOKEN=$(echo "$CREDS" | jq -r '.refresh_token')

[[ -z "$REFRESH_TOKEN" || "$REFRESH_TOKEN" == "null" ]] && {
    echo "❌ refresh_token not found. Run: gws auth login --scopes 'https://www.googleapis.com/auth/drive.readonly'"
    exit 1
}

# 2. Exchange refresh token for access token
echo "→ Exchanging refresh token for access token..."
TOKEN_RESPONSE=$(curl -s -X POST https://oauth2.googleapis.com/token \
    -d "client_id=${CLIENT_ID}" \
    -d "client_secret=${CLIENT_SECRET}" \
    -d "refresh_token=${REFRESH_TOKEN}" \
    -d "grant_type=refresh_token")

ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.access_token // empty')
[[ -z "$ACCESS_TOKEN" ]] && {
    echo "❌ Failed to get access token. Response:"
    echo "$TOKEN_RESPONSE" | head -10
    exit 1
}
echo "✅ Got access token (expires in ~1 hour)"

# 3. Resolve folder ID
echo
echo "→ Resolving folder ID for '$FOLDER_NAME'..."
FOLDER_ID=$(gws_json drive files list \
    --params "$(jq -nc --arg q "name='$FOLDER_NAME' and mimeType='application/vnd.google-apps.folder' and trashed=false" \
              '{q: $q, fields: "files(id)"}')" \
    | jq -r '.files[0].id')
echo "✅ Folder ID: $FOLDER_ID"

# 4. List files
FILES=$(gws_json drive files list \
    --params "$(jq -nc --arg q "parents='$FOLDER_ID' and trashed=false" \
              '{q: $q, pageSize: 1000, fields: "files(id,name,mimeType,size,md5Checksum)"}')")

# 5. For each application/json file, curl directly from Drive API
echo
echo "→ Re-downloading application/json files via curl..."
mkdir -p "$LOCAL_DIR"

# Use process substitution to avoid subshell var loss
while IFS= read -r entry; do
    fid=$(echo "$entry" | jq -r '.id')
    fname=$(echo "$entry" | jq -r '.name')
    fsize=$(echo "$entry" | jq -r '.size')
    fmd5=$(echo "$entry" | jq -r '.md5Checksum')
    dest="$LOCAL_DIR/$fname"

    echo "  ↓ $fname ($fsize bytes)"

    # Direct Drive REST API call with alt=media — bypasses gws entirely
    if ! curl -sL \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Accept: */*" \
        "https://www.googleapis.com/drive/v3/files/${fid}?alt=media" \
        -o "$dest"; then
        echo "    ❌ curl failed"
        continue
    fi

    actual_size=$(stat -f%z "$dest" 2>/dev/null || echo 0)
    actual_md5=$(md5 -q "$dest")

    if [[ "$actual_size" == "$fsize" ]] && [[ "$actual_md5" == "$fmd5" ]]; then
        echo "    ✅ size $actual_size, md5 match"
    else
        echo "    ⚠ size=$actual_size (expected $fsize), md5=$actual_md5 (expected $fmd5)"
    fi
done < <(echo "$FILES" | jq -c '.files[] | select(.mimeType == "application/json")')

echo
echo "→ Final state:"
ls -lh "$LOCAL_DIR"
echo
echo "→ Total size:"
du -sh "$LOCAL_DIR"
