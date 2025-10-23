#!/bin/sh

# Download small Socrata datasets for testing (skips files > 100MB)
# Usage: download_small_datasets.sh [app token file]
# Requires curl and jq.

mkdir -p datasets && cd datasets || exit 1

app_token=$([ -f "$1" ] && head -1 "$1")
discovery_api_url='https://api.us.socrata.com/api/catalog/v1'
scrollid=
count=0
max_datasets=10
max_size_mb=100
max_size_bytes=$((max_size_mb * 1024 * 1024))

echo "Downloading ${max_datasets} small datasets (max size: ${max_size_mb}MB)..."
echo "================================================"

while [ $count -lt $max_datasets ]; do
    echo "Requesting API (downloaded: ${count}/${max_datasets})..."
    
    res=$(curl --no-progress-meter -f -G "$discovery_api_url" \
        -d only=datasets \
        -d provenance=official \
        -d limit=20 \
        -d scroll_id="$scrollid" \
        -H "X-App-Token: $app_token") || {
        echo "API request failed, retrying..."
        continue
    }

    if [ "$(printf '%s' "$res" | jq '.results | length')" -eq 0 ]; then
        echo "No more datasets available"
        break
    fi
    
    scrollid=$(printf '%s' "$res" | jq -r '.results[-1].resource.id')

    # Process results without subshell by using process substitution
    while IFS= read -r resource; do
        if [ $count -ge $max_datasets ]; then
            break
        fi
        
        id=$(printf '%s' "$resource" | jq -r '.resource.id')
        name=$(printf '%s' "$resource" | jq -r '.resource.name')
        domain=$(printf '%s' "$resource" | jq -r '.metadata.domain')
        download_url="$domain/api/views/$id/rows.csv?accessType=DOWNLOAD"
        
        echo "[$((count + 1))/${max_datasets}] Downloading: $id - $name"
        
        mkdir -p "$id"
        printf '%s' "$resource" | jq '.' > "$id/metadata.json"
        
        # Use timeout (30 seconds) to prevent hanging on large files
        # Approximate download speed check: 100MB in 30s = ~3.3MB/s minimum
        if timeout 30s curl --no-progress-meter -fL -o "$id/rows.csv" "$download_url" 2>/dev/null; then
            # Check file size after download
            file_size=$(stat -f%z "$id/rows.csv" 2>/dev/null || stat -c%s "$id/rows.csv" 2>/dev/null)
            if [ "$file_size" -gt "$max_size_bytes" ]; then
                size_mb=$((file_size / 1024 / 1024))
                echo "  ⊘ Too large (${size_mb}MB), skipping"
                rm -rf "$id"
                continue
            fi
            
            actual_size=$(du -h "$id/rows.csv" | cut -f1)
            echo "  ✓ Download successful (size: ${actual_size})"
            count=$((count + 1))
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo "  ⊘ Download timeout (likely too large), skipping"
            else
                echo "  ✗ Download failed, skipping"
            fi
            rm -rf "$id"
            continue
        fi
        
        if [ $count -ge $max_datasets ]; then
            break
        fi
    done < <(printf '%s' "$res" | jq -c '.results[]')
    
    if [ $count -ge $max_datasets ]; then
        break
    fi
done

cd ..
echo "================================================"
echo "✓ Complete! Downloaded ${count} datasets to datasets/ directory"
echo "You can view them with:"
echo "  ls -lh datasets/"
echo "  du -sh datasets/*/"

