import msal
import requests
import pandas as pd


class DataverseConnector:
    def __init__(self, tenant_id, client_id, client_secret, org_url):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.org_url = org_url.rstrip("/")
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"
        self.token = self._get_token()

    def _get_token(self):
        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=self.authority,
            client_credential=self.client_secret,
        )
        token_result = app.acquire_token_for_client(
            scopes=[f"{self.org_url}/.default"]
        )
        if "access_token" not in token_result:
            raise Exception(
                f"Authentication failed: {token_result.get('error_description')}"
            )
        return token_result["access_token"]

    def get_tables(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        url = f"{self.org_url}/api/data/v9.2/EntityDefinitions?$select=LogicalName"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("value", [])
        return [t["LogicalName"] for t in data]

    def get_fields(self, table_name):
        headers = {"Authorization": f"Bearer {self.token}"}
        url = (
            f"{self.org_url}/api/data/v9.2/EntityDefinitions(LogicalName='{table_name}')"
            f"/Attributes?$select=LogicalName,DisplayName"
        )
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("value", [])

        fields = []
        for field in data:
            display_name = field.get("DisplayName", {}).get("UserLocalizedLabel", {}).get(
                "Label", ""
            )
            fields.append({"LogicalName": field["LogicalName"], "DisplayName": display_name})
        return fields

    def get_table(self, table_name, select="*", top=5000):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0",
        }

        if not select or select == "*":
            url = f"{self.org_url}/api/data/v9.2/{table_name}?$top={top}"
        else:
            url = f"{self.org_url}/api/data/v9.2/{table_name}?$select={select}&$top={top}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json().get("value", [])
        df = pd.DataFrame(data)

        # Clean up system/lookup columns
        df = self.clean_system_columns(df)

        # Ensure unique column names
        df = self.make_columns_unique(df)

        return df

    @staticmethod
    def clean_system_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [
            col for col in df.columns if col.startswith("@odata.") or col.endswith("_value") or col.endswith("_guid")
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors="ignore")
        return df

    @staticmethod
    def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            dups = cols[cols == dup].index.tolist()
            for i, idx in enumerate(dups[1:], start=1):
                cols[idx] = f"{dup}_{i}"
        df.columns = cols
        return df
