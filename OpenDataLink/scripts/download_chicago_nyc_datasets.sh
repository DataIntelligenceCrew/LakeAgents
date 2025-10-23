#!/bin/sh

# Download datasets from Chicago and NYC
# Downloads 25 datasets from Chicago and 25 from NYC
# Skips datasets smaller than 5MB
# Requires curl and jq

cd /localdisk3/ytang49/opendata/OpenDataLink || exit 1
mkdir -p datasets && cd datasets || exit 1

discovery_api_url='https://api.us.socrata.com/api/catalog/v1'
min_size_mb=5  # Minimum file size in MB (skip datasets smaller than this)
min_size_bytes=$((min_size_mb * 1024 * 1024))

# Function to download datasets from a specific domain
download_from_domain() {
    local domain="$1"
    local target_count="$2"
    local city_name="$3"
    local count=0
    local scrollid=""
    
    echo "==== Downloading $target_count datasets from $city_name ($domain) ===="
    
    while [ $count -lt $target_count ]; do
        # Query API for datasets from specific domain
        res=$(curl --no-progress-meter -f -G "$discovery_api_url" \
            -d domains="$domain" \
            -d only=datasets \
            -d provenance=official \
            -d limit=50 \
            -d scroll_id="$scrollid" 2>/dev/null) || {
            echo "API request failed, retrying..."
            sleep 2
            continue
        }
        
        # Check if we got results
        result_count=$(printf '%s' "$res" | jq '.results | length' 2>/dev/null)
        if [ -z "$result_count" ] || [ "$result_count" -eq 0 ]; then
            echo "No more results from $city_name"
            break
        fi
        
        scrollid=$(printf '%s' "$res" | jq -r '.results[-1].resource.id')
        
        # Process each dataset
        while IFS= read -r resource; do
            if [ $count -ge $target_count ]; then
                break
            fi
            
            id=$(printf '%s' "$resource" | jq -r '.resource.id')
            
            # Skip if already downloaded
            if [ -d "$id" ] && [ -f "$id/rows.csv" ]; then
                echo "[$city_name $((count + 1))/$target_count] Dataset $id already exists, skipping"
                continue
            fi
            
            echo "[$city_name $((count + 1))/$target_count] Downloading dataset $id"
            
            mkdir -p "$id"
            printf '%s' "$resource" | jq '.' > "$id/metadata.json"
            
            domain_url=$(printf '%s' "$resource" | jq -r '.metadata.domain')
            download_url="https://$domain_url/api/views/$id/rows.csv?accessType=DOWNLOAD"
            
            # Download without size or time limits
            if curl --no-progress-meter -fL -o "$id/rows.csv" "$download_url" 2>/dev/null; then
                if [ -f "$id/rows.csv" ]; then
                    file_size=$(stat -f%z "$id/rows.csv" 2>/dev/null || stat -c%s "$id/rows.csv" 2>/dev/null)
                    
                    if [ "$file_size" -lt 100 ]; then
                        echo "  ✗ File empty or error, removing"
                        rm -rf "$id"
                        continue
                    fi
                    
                    # Check minimum size requirement
                    if [ "$file_size" -lt "$min_size_bytes" ]; then
                        echo "  ✗ File too small ($(($file_size / 1024 / 1024))MB < ${min_size_mb}MB), skipping"
                        rm -rf "$id"
                        continue
                    fi
                    
                    echo "  ✓ Downloaded successfully ($(($file_size / 1024 / 1024))MB)"
                    count=$((count + 1))
                else
                    echo "  ✗ Download failed, removing directory"
                    rm -rf "$id"
                fi
            else
                echo "  ✗ Download failed"
                rm -rf "$id"
            fi
            
        done < <(printf '%s' "$res" | jq -c '.results[]')
        
        if [ $count -ge $target_count ]; then
            break
        fi
        
        sleep 1
    done
    
    echo "==== Completed: Downloaded $count datasets from $city_name ===="
    echo ""
}

# Download from Chicago
download_from_domain "data.cityofchicago.org" 25 "Chicago"

# Download from NYC
download_from_domain "data.cityofnewyork.us" 25 "NYC"

echo "==== All downloads complete! ===="
echo "Total datasets downloaded:"
ls -d */ | wc -l

