#!/usr/bin/env python3
"""
Search and download Socrata datasets by name/domain
Usage: python download_byname.py <search_term> [--download] [--max N] [--domain DOMAIN]
"""

import json
import sys
import requests
from pathlib import Path
from typing import List, Dict, Optional


def search_socrata_datasets(
    search_term: str,
    max_results: int = 50,
    app_token: Optional[str] = None,
    domain_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Search for datasets on Socrata Discovery API by name/keyword
    
    Args:
        search_term: The term to search for in dataset names and descriptions
        max_results: Maximum number of results to return
        app_token: Socrata app token (optional, but recommended)
        
    Returns:
        List of dictionaries containing dataset information
    """
    discovery_api_url = 'https://api.us.socrata.com/api/catalog/v1'
    
    headers = {}
    if app_token:
        headers['X-App-Token'] = app_token
    
    params = {
        'q': search_term,
        'only': 'datasets',
        'provenance': 'official',
        'limit': min(max_results, 100)  # API limit is 100 per request
    }
    if domain_filter:
        params['domains'] = domain_filter
    
    try:
        if domain_filter:
            print(f"Searching Socrata for: '{search_term}' on domain '{domain_filter}'...")
        else:
            print(f"Searching Socrata for: '{search_term}'...")
        response = requests.get(discovery_api_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get('results', []):
            resource = item.get('resource', {})
            metadata = item.get('metadata', {})
            
            results.append({
                'id': resource.get('id'),
                'name': resource.get('name'),
                'description': resource.get('description', ''),
                'domain': metadata.get('domain', ''),
                'permalink': item.get('permalink', ''),
                'link': item.get('link', ''),
                'columns': resource.get('columns_name', []),
                'columns_count': len(resource.get('columns_name', [])),
                'created_at': resource.get('createdAt', ''),
                'updated_at': resource.get('updatedAt', ''),
                'download_count': resource.get('download_count', 0),
                'page_views_total': resource.get('page_views', {}).get('page_views_total', 0),
                'full_metadata': item  # Store full metadata for download
            })
        
        if domain_filter:
            domain_lower = domain_filter.lower()
            results = [
                r for r in results
                if r.get('domain', '').lower() == domain_lower
            ]

        return results
        
    except requests.RequestException as e:
        print(f"Error searching Socrata API: {e}", file=sys.stderr)
        return []


def get_dataset_by_id(dataset_id: str, domain: str = None, app_token: str = None) -> Dict:
    """
    Get dataset metadata directly by ID
    
    Args:
        dataset_id: The Socrata dataset ID (four-by-four format)
        domain: Optional domain hint (e.g., 'data.cityofnewyork.us')
        app_token: Socrata app token (optional)
        
    Returns:
        Dataset dictionary or None if not found
    """
    # Try to get metadata from Discovery API
    discovery_api_url = 'https://api.us.socrata.com/api/catalog/v1'
    
    headers = {}
    if app_token:
        headers['X-App-Token'] = app_token
    
    # First, try to search for the ID
    params = {
        'ids': dataset_id,
        'only': 'datasets'
    }
    
    try:
        print(f"Fetching dataset metadata for ID: {dataset_id}...")
        response = requests.get(discovery_api_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('results', [])
        
        if results:
            item = results[0]
            resource = item.get('resource', {})
            metadata = item.get('metadata', {})
            
            return {
                'id': resource.get('id'),
                'name': resource.get('name'),
                'description': resource.get('description', ''),
                'domain': metadata.get('domain', ''),
                'permalink': item.get('permalink', ''),
                'link': item.get('link', ''),
                'columns': resource.get('columns_name', []),
                'columns_count': len(resource.get('columns_name', [])),
                'created_at': resource.get('createdAt', ''),
                'updated_at': resource.get('updatedAt', ''),
                'download_count': resource.get('download_count', 0),
                'page_views_total': resource.get('page_views', {}).get('page_views_total', 0),
                'full_metadata': item
            }
        
        # If not found via Discovery API, try direct domain API
        if domain:
            try:
                domain_api_url = f"https://{domain}/api/views/{dataset_id}.json"
                print(f"Trying direct API: {domain_api_url}...")
                response = requests.get(domain_api_url, headers=headers, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                # Construct a minimal dataset dict from direct API response
                return {
                    'id': dataset_id,
                    'name': data.get('name', 'Unknown'),
                    'description': data.get('description', ''),
                    'domain': domain,
                    'permalink': f"https://{domain}/d/{dataset_id}",
                    'link': f"https://{domain}/d/{dataset_id}",
                    'columns': [col.get('name', '') for col in data.get('columns', [])],
                    'columns_count': len(data.get('columns', [])),
                    'created_at': data.get('createdAt', ''),
                    'updated_at': data.get('updatedAt', ''),
                    'download_count': 0,
                    'page_views_total': 0,
                    'full_metadata': data
                }
            except:
                pass
        
        return None
        
    except requests.RequestException as e:
        print(f"Error fetching dataset: {e}", file=sys.stderr)
        return None


def download_dataset(dataset: Dict, output_dir: str = "datasets") -> bool:
    """
    Download a dataset from Socrata
    
    Args:
        dataset: Dataset dictionary from search results
        output_dir: Directory to save the dataset
        
    Returns:
        True if successful, False otherwise
    """
    dataset_id = dataset['id']
    domain = dataset['domain']
    
    if not dataset_id or not domain:
        print(f"Error: Missing dataset ID or domain for {dataset['name']}", file=sys.stderr)
        return False
    
    # Create output directory
    output_path = Path(output_dir) / dataset_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata_file = output_path / "metadata.json"
    try:
        with open(metadata_file, 'w') as f:
            json.dump(dataset['full_metadata'], f, indent=2)
        print(f"  ✓ Saved metadata to {metadata_file}")
    except Exception as e:
        print(f"  ✗ Error saving metadata: {e}", file=sys.stderr)
        return False
    
    # Download CSV data
    csv_file = output_path / "rows.csv"
    download_url = f"https://{domain}/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD"
    
    try:
        print(f"  Downloading data from {download_url}...")
        response = requests.get(download_url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(csv_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Get file size
        file_size = csv_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"  ✓ Downloaded {size_mb:.2f} MB to {csv_file}")
        return True
        
    except requests.RequestException as e:
        print(f"  ✗ Error downloading data: {e}", file=sys.stderr)
        return False


def print_results(results: List[Dict], verbose: bool = False, show_all_columns: bool = False):
    """Print search results in a readable format"""
    if not results:
        print("\nNo datasets found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(results)} dataset(s):")
    print(f"{'='*80}\n")
    
    for i, dataset in enumerate(results, 1):
        print(f"[{i}] {dataset['name']}")
        print(f"    ID: {dataset['id']}")
        print(f"    Domain: {dataset['domain']}")
        
        if verbose:
            print(f"    Link: {dataset['link']}")
            print(f"    Downloads: {dataset.get('download_count', 0):,}")
            print(f"    Page Views: {dataset.get('page_views_total', 0):,}")
            print(f"    Created: {dataset.get('created_at', 'N/A')[:10]}")
            print(f"    Updated: {dataset.get('updated_at', 'N/A')[:10]}")
            
            if dataset['description']:
                desc = dataset['description'][:300]
                if len(dataset['description']) > 300:
                    desc += "..."
                # Remove HTML tags
                import re
                desc = re.sub('<[^<]+?>', '', desc)
                print(f"    Description: {desc}")
            
            if dataset['columns']:
                if show_all_columns:
                    print(f"    Columns ({len(dataset['columns'])}):")
                    for col in dataset['columns']:
                        print(f"      - {col}")
                else:
                    print(f"    Columns ({len(dataset['columns'])}): {', '.join(dataset['columns'][:8])}")
                    if len(dataset['columns']) > 8:
                        print(f"             ... and {len(dataset['columns']) - 8} more")
        else:
            print(f"    Columns: {dataset.get('columns_count', 0)}")
        
        print()


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python download_byname.py <search_term> [options]")
        print("\nOptions:")
        print("  --download           Download selected datasets")
        print("  --download-all       Download all search results")
        print("  --max N              Maximum number of results (default: 50)")
        print("  --domain DOMAIN      Restrict search to a specific domain (e.g. data.cityofnewyork.us)")
        print("  --output DIR         Output directory (default: ../datasets)")
        print("  --verbose, -v        Show detailed information")
        print("  --has-column NAME    Filter: dataset must contain a column matching NAME (repeatable)")
        print("  --require-all-columns Require all --has-column names to match (default: any)")
        print("  --token FILE         Socrata app token file")
        print("  --id DATASET_ID      Download dataset directly by ID (e.g., gzfs-3h4m)")
        print("  --show-columns       Print all column names for each dataset")
        print("\nExamples:")
        print("  python download_byname.py 'agency spending'")
        print("  python download_byname.py --id gzfs-3h4m --domain data.cityofnewyork.us")
        print("  python download_byname.py 'taxi' --domain data.cityofnewyork.us --download --max 10")
        print("  python download_byname.py 'spending' --domain data.cityofnewyork.us --has-column 'Agency Name'")
        print("  python download_byname.py 'budget' --has-column 'Agency' --has-column 'Expenditures' --require-all-columns --show-columns")
        print("  python download_byname.py 'crime' --download-all --output datasets/")
        print("  python download_byname.py 'traffic' --verbose")
        sys.exit(1)
    
    search_term = None
    dataset_id = None
    download = "--download" in sys.argv
    download_all = "--download-all" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    show_all_columns = "--show-columns" in sys.argv
    max_results = 50
    output_dir = "../datasets"
    app_token = None
    domain_filter: Optional[str] = None
    domain_hint = None
    # Column filtering
    required_columns: List[str] = []
    require_all_columns = "--require-all-columns" in sys.argv
    
    # Check if --id is used (must be first or second argument)
    if len(sys.argv) > 1 and sys.argv[1] == "--id":
        if len(sys.argv) < 3:
            print("Error: --id requires a dataset ID", file=sys.stderr)
            sys.exit(1)
        dataset_id = sys.argv[2]
        # Parse remaining arguments starting from index 3
        arg_start = 3
    else:
        # Normal search mode - first arg is search term
        if len(sys.argv) < 2:
            print("Error: Please provide a search term or use --id to download by dataset ID", file=sys.stderr)
            sys.exit(1)
        search_term = sys.argv[1]
        arg_start = 2
    
    # Parse optional arguments
    for i, arg in enumerate(sys.argv[arg_start:], start=arg_start):
        if arg == "--max" and i + 1 < len(sys.argv):
            max_results = int(sys.argv[i + 1])
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
        elif arg == "--token" and i + 1 < len(sys.argv):
            token_file = sys.argv[i + 1]
            try:
                with open(token_file, 'r') as f:
                    app_token = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read token file: {e}", file=sys.stderr)
        elif arg == "--domain" and i + 1 < len(sys.argv):
            domain_value = sys.argv[i + 1].strip()
            # Check if this is for --id or for search filter
            if dataset_id:
                domain_hint = domain_value
            else:
                domain_filter = domain_value
        elif arg == "--has-column" and i + 1 < len(sys.argv):
            required_columns.append(sys.argv[i + 1].strip())
    
    # If --id is specified, download directly by ID
    if dataset_id:
        dataset = get_dataset_by_id(dataset_id, domain_hint, app_token)
        if not dataset:
            print(f"Error: Could not find dataset with ID '{dataset_id}'", file=sys.stderr)
            if not domain_hint:
                print("Hint: Try specifying --domain to help locate the dataset", file=sys.stderr)
            sys.exit(1)
        
        print(f"\n{'='*80}")
        print(f"Dataset found: {dataset['name']}")
        print(f"{'='*80}\n")
        print_results([dataset], verbose, show_all_columns)
        
        print(f"\n{'='*80}")
        print(f"Downloading dataset to {output_dir}...")
        print(f"{'='*80}\n")
        
        if download_dataset(dataset, output_dir):
            print(f"\n{'='*80}")
            print(f"✓ Successfully downloaded dataset!")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print(f"✗ Failed to download dataset")
            print(f"{'='*80}")
            sys.exit(1)
        
        sys.exit(0)
    
    # Otherwise, search by term (search_term should already be set; allow empty string)
    if search_term is None:
        print("Error: Please provide a search term or use --id to download by dataset ID", file=sys.stderr)
        sys.exit(1)
    
    # Search for datasets on Socrata
    results = search_socrata_datasets(
        search_term,
        max_results,
        app_token=app_token,
        domain_filter=domain_filter,
    )
    
    if not results:
        print("\nNo datasets found. Try a different search term.")
        sys.exit(0)
    
    # Filter by required columns if specified (case-insensitive substring match)
    if required_columns:
        req_norm = [c.lower() for c in required_columns]
        def columns_match(cols: List[str]) -> bool:
            col_norm = [c.lower() for c in (cols or [])]
            if require_all_columns:
                return all(any(req in col for col in col_norm) for req in req_norm)
            else:
                return any(any(req in col for col in col_norm) for req in req_norm)
        before = len(results)
        results = [r for r in results if columns_match(r.get('columns', []))]
        after = len(results)
        print(f"\nFiltered by columns ({'ALL' if require_all_columns else 'ANY'} match on {len(required_columns)} name(s)): {before} -> {after}")
    
    # Print results
    print_results(results, verbose, show_all_columns)
    
    # Download datasets if requested
    if download_all:
        print(f"\n{'='*80}")
        print(f"Downloading all {len(results)} datasets to {output_dir}...")
        print(f"{'='*80}\n")
        
        success_count = 0
        for i, dataset in enumerate(results, 1):
            print(f"[{i}/{len(results)}] {dataset['name']} ({dataset['id']})")
            if download_dataset(dataset, output_dir):
                success_count += 1
            print()
        
        print(f"\n{'='*80}")
        print(f"Successfully downloaded {success_count}/{len(results)} datasets")
        print(f"{'='*80}")
        
    elif download:
        print(f"\n{'='*80}")
        print("Enter dataset numbers to download (comma-separated), or 'all' for all:")
        print("Example: 1,3,5  or  all")
        print(f"{'='*80}")
        
        try:
            user_input = input("\nYour choice: ").strip()
            
            if user_input.lower() == 'all':
                indices = list(range(len(results)))
            else:
                indices = [int(x.strip()) - 1 for x in user_input.split(',')]
                indices = [i for i in indices if 0 <= i < len(results)]
            
            if not indices:
                print("No valid selections made.")
                sys.exit(0)
            
            print(f"\n{'='*80}")
            print(f"Downloading {len(indices)} dataset(s) to {output_dir}...")
            print(f"{'='*80}\n")
            
            success_count = 0
            for idx in indices:
                dataset = results[idx]
                print(f"[{indices.index(idx)+1}/{len(indices)}] {dataset['name']} ({dataset['id']})")
                if download_dataset(dataset, output_dir):
                    success_count += 1
                print()
            
            print(f"\n{'='*80}")
            print(f"Successfully downloaded {success_count}/{len(indices)} datasets")
            print(f"{'='*80}")
            
        except (ValueError, KeyboardInterrupt) as e:
            print("\nDownload cancelled.")
            sys.exit(0)


if __name__ == "__main__":
    main()