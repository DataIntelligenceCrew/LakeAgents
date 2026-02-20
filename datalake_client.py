"""
Socrata Datalake Client for connecting to opendata repository
"""
import requests
from typing import Dict, Any, List, Optional


class SocrataDatalakeClient:
    def __init__(self, config: Dict[str, Any]):
        self.discovery_api_url = 'https://api.us.socrata.com/api/catalog/v1'
        self.app_token = config.get('app_token')
        self.api_key_id = config.get('api_key_id')           
        self.api_key_secret = config.get('api_key_secret')   
        self.domains = config.get('domains', [])
        self.max_tables = config.get('max_tables', None)
        self.headers = {}
        if self.app_token:
            self.headers['X-App-Token'] = self.app_token
        self.auth = None
        if self.api_key_id and self.api_key_secret:
            self.auth = (self.api_key_id, self.api_key_secret)

    def read_metadata(
        self,
        dataset_name: Optional[str] = None,
        exclude_tables: List[str] = None,
        search_domains: Optional[List[str]] = None,
        search_q: Optional[str] = None,
        limit: int = 1,
        offset: int = 0,
    ) -> Dict[str, Any]:
        params = {
            'only': 'datasets',
            'provenance': 'official',
            'limit': limit,
            'offset': offset,
        }
        domains = search_domains if search_domains is not None else self.domains
        if domains:
            params['domains'] = ','.join(domains)
        if search_q and search_q.strip():
            params['q'] = search_q.strip()

        exclude_set = set()
        if exclude_tables:
            exclude_set = {t.lower().strip() for t in exclude_tables}

        try:
            response = requests.get(
                self.discovery_api_url,
                params=params,
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            return {"metadata_by_dataset": {}, "errors": {"api": str(e)}}

        raw_results = data.get('results', [])
        out: Dict[str, Any] = {}
        for item in raw_results:
            resource = item.get('resource', {})
            ds_id = resource.get('id')
            if not ds_id or ds_id.lower().strip() in exclude_set:
                continue
            out[ds_id] = {"table_description": resource.get("description", "")}
        n = len(out)
        print(f"[read_metadata] returned {n} candidates: {list(out.keys())[:5]}...")
        return {
            "metadata_by_dataset": out,
            "errors": {},
            "raw_results_count": len(raw_results),
            "resultSetSize": data.get("resultSetSize"),
        }

    def get_dataset_metadata(self, dataset_id: str, domain: str) -> Dict[str, Any]:
        """Fetch full metadata for one dataset (id, description, attribution, columns, classification, domain)."""
        url = f"https://{domain.rstrip('/')}/api/views/{dataset_id}.json"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def read_data(self, dataset_id: str, domain: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        domain = domain.rstrip('/')
        url = f"https://{domain}/resource/{dataset_id}.json"
        params = {}
        if max_rows:
            params["$limit"] = max_rows
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                auth=self.auth,   
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return []