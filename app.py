# BUSINESS SCIENCE
# Pandas Data Analyst App
# -----------------------

from openai import OpenAI
from dataverse_connector import DataverseConnector
import streamlit as st
import pandas as pd
import plotly.io as pio
import json
import requests

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from ai_data_science_team import (
    PandasDataAnalyst,
    DataWranglingAgent,
    DataVisualizationAgent,
)

# ---------------------------
# Helper functions: Dataverse metadata and auth
# ---------------------------

@st.cache_data
def get_access_token(tenant_id: str, client_id: str, client_secret: str, resource_url: str) -> str:
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    scope = f"{resource_url.rstrip('/')}/.default"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
    }
    resp = requests.post(token_url, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]

@st.cache_data
def get_tables(access_token: str, org_url: str):
    """Fetch only custom tables from Dataverse, return EntitySetName + DisplayName"""
    endpoint = (
        f"{org_url.rstrip('/')}/api/data/v9.2/EntityDefinitions"
        "?$select=EntitySetName,DisplayName,IsCustomEntity"
    )
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    resp = requests.get(endpoint, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = []
    for t in resp.json().get("value", []):
        if not t.get("IsCustomEntity", False):
            continue  # skip system tables
        labels = t.get("DisplayName", {}).get("LocalizedLabels")
        display = labels[0]["Label"] if labels and len(labels) > 0 else t["EntitySetName"]
        tables.append({"EntitySetName": t["EntitySetName"], "DisplayName": display})
    return tables

@st.cache_data
def get_fields(table_name: str, access_token: str, org_url: str):
    """
    Fetch fields/attributes for a given Dataverse table.
    Tries both singular and plural forms, and filters out problematic system fields.
    """
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    # Try original table_name first
    endpoint = f"{org_url.rstrip('/')}/api/data/v9.2/EntityDefinitions(LogicalName='{table_name}')/Attributes?$select=LogicalName,DisplayName,AttributeType,IsPrimaryId,IsCustomAttribute"
    resp = requests.get(endpoint, headers=headers, timeout=30)

    # If 404, attempt singular form
    if resp.status_code == 404 and table_name.endswith("s"):
        singular_name = table_name[:-1]
        endpoint = f"{org_url.rstrip('/')}/api/data/v9.2/EntityDefinitions(LogicalName='{singular_name}')/Attributes?$select=LogicalName,DisplayName,AttributeType,IsPrimaryId,IsCustomAttribute"
        resp = requests.get(endpoint, headers=headers, timeout=30)

    if not resp.ok:
        st.error(f"Failed to fetch fields for {table_name}: {resp.status_code} - {resp.text}")
        return []

    fields = []
    for f in resp.json().get("value", []):
        attr_type = f.get("AttributeType", "")
        logical_name = f.get("LogicalName", "")
        is_primary = f.get("IsPrimaryId", False)
        is_custom = f.get("IsCustomAttribute", False)
        
        # Skip problematic system attribute types, but keep primary ID and custom fields
        if attr_type in ["Virtual", "CalcRollup"] and not is_primary:
            continue
        
        # Skip common system fields that cause issues, but keep primary and custom fields
        system_fields_to_skip = [
            "owneridname", "owneridyominame", "owneridtype", 
            "createdbyname", "createdbyyominame", "modifiedbyname", "modifiedbyyominame",
            "createdonbehalfbyname", "createdonbehalfbyyominame", "modifiedonbehalfbyname", "modifiedonbehalfbyyominame",
            "owningbusinessunitname", "transactioncurrencyidname"
        ]
        
        if logical_name in system_fields_to_skip and not is_custom:
            continue
            
        labels = f.get("DisplayName", {}).get("LocalizedLabels")
        display = labels[0]["Label"] if labels and len(labels) > 0 else logical_name
        
        fields.append({
            "LogicalName": logical_name, 
            "DisplayName": display,
            "AttributeType": attr_type,
            "IsPrimaryId": is_primary,
            "IsCustomAttribute": is_custom
        })
    
    # Sort fields: Primary ID first, then custom fields, then system fields
    fields.sort(key=lambda x: (not x["IsPrimaryId"], not x["IsCustomAttribute"], x["LogicalName"]))
    return fields

def test_field_access(table_name: str, fields: list, access_token: str, org_url: str) -> list:
    """
    Test field accessibility by trying different combinations, starting with safe fields.
    """
    if not fields:
        return []
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0"
    }
    
    # First, try to identify the primary key and some basic safe fields
    primary_field = next((f for f in fields if f.get("IsPrimaryId", False)), None)
    custom_fields = [f for f in fields if f.get("IsCustomAttribute", False)]
    
    # Start with a minimal test - just primary key if available
    if primary_field:
        test_url = f"{org_url.rstrip('/')}/api/data/v9.2/{table_name}?$select={primary_field['LogicalName']}&$top=1"
        try:
            resp = requests.get(test_url, headers=headers, timeout=30)
            if resp.ok:
                st.success(f"✓ Primary key field '{primary_field['LogicalName']}' is accessible")
                
                # Now test with a few custom fields if they exist
                if custom_fields:
                    test_fields = [primary_field["LogicalName"]] + [f["LogicalName"] for f in custom_fields[:3]]
                    test_select = ",".join(test_fields)
                    test_url = f"{org_url.rstrip('/')}/api/data/v9.2/{table_name}?$select={test_select}&$top=1"
                    
                    resp = requests.get(test_url, headers=headers, timeout=30)
                    if resp.ok:
                        st.success("✓ Custom fields are accessible")
                        return fields  # All fields should be accessible
                    else:
                        st.warning("⚠️ Some custom fields may have issues. Testing individual fields...")
                        # Test each custom field individually
                        accessible_fields = [primary_field]
                        for custom_field in custom_fields:
                            test_url = f"{org_url.rstrip('/')}/api/data/v9.2/{table_name}?$select={primary_field['LogicalName']},{custom_field['LogicalName']}&$top=1"
                            resp = requests.get(test_url, headers=headers, timeout=30)
                            if resp.ok:
                                accessible_fields.append(custom_field)
                            else:
                                st.warning(f"⚠️ Field '{custom_field['LogicalName']}' is not accessible")
                        return accessible_fields
                else:
                    return [primary_field]  # At least return the primary field
            else:
                st.error(f"❌ Cannot access primary field. Status: {resp.status_code}")
                st.error(f"Response: {resp.text}")
        except Exception as e:
            st.error(f"❌ Error testing primary field access: {e}")
    
    # If primary field test failed, try with basic system fields that usually exist
    basic_system_fields = [f for f in fields if f["LogicalName"] in ["createdon", "modifiedon", "statecode", "statuscode"]]
    if basic_system_fields:
        st.info("Trying basic system fields...")
        return basic_system_fields[:3]
    
    # Last resort - return first few fields
    st.warning("Falling back to first few fields")
    return fields[:3]

def apply_display_name_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col in mapping:
            rename_map[col] = mapping[col]
        else:
            base = col
            for suffix in ["@OData.Community.Display.V1.FormattedValue", "_value", "_guid"]:
                if base.endswith(suffix):
                    base = base.replace(suffix, "")
            if base in mapping:
                rename_map[col] = mapping[base]
    return df.rename(columns=rename_map) if rename_map else df

def clean_system_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove Dataverse system/lookup columns that are automatically included
    but usually not needed for analysis.
    """
    cols_to_drop = [col for col in df.columns if col.startswith("@odata.") or col.endswith("_value") or col.endswith("_guid")]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

# ---------------------------
# App Inputs / Config
# ---------------------------
MODEL_LIST = ["gpt-4o-mini", "gpt-4o"]
TITLE = "Pandas Data Analyst AI Copilot"

st.set_page_config(page_title=TITLE, page_icon="📊")
st.title(TITLE)

st.markdown("""
Welcome to the Pandas Data Analyst AI. Upload a CSV/Excel file or connect to Dataverse and ask questions about your data.  
The AI agent will analyze your dataset and return either data tables or interactive charts.
""")

with st.expander("Example Questions", expanded=False):
    st.write("""
        ##### Bikes Data Set:
        - Show the top 5 bike models by extended sales.
        - Show the top 5 bike models by extended sales in a bar chart.
        - Show the top 5 bike models by extended sales in a pie chart.
        - Make a plot of extended sales by month for each bike model. Use a color to identify the bike models.
    """)

# ---------------------------
# OpenAI API Key Input
# ---------------------------
st.sidebar.header("Enter your OpenAI API Key")
st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input(
    "API Key",
    type="default",  # visible input, avoids browser password autofill
    help="Your OpenAI API key is required for the app to function."
)

if not st.session_state["OPENAI_API_KEY"]:
    st.info("Please enter your OpenAI API Key to proceed.")
    st.stop()

# Validate API key
client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])
try:
    client.models.list()
    st.success("API Key is valid!")
except Exception as e:
    st.error(f"Invalid API Key: {e}")
    st.stop()

# ---------------------------
# OpenAI Model Selection
# ---------------------------
model_option = st.sidebar.selectbox("Choose OpenAI model", MODEL_LIST, index=0)
llm = ChatOpenAI(model=model_option, api_key=st.session_state["OPENAI_API_KEY"])

# ---------------------------
# Data Source Selection
# ---------------------------
st.sidebar.header("Select Data Source")
data_source = st.sidebar.radio("Choose data source:", ("Upload CSV/Excel", "Dataverse"), index=0)

if "df" not in st.session_state: st.session_state["df"] = None
if "dv_field_map" not in st.session_state: st.session_state["dv_field_map"] = {}

# --- CSV/Excel Upload ---
if data_source == "Upload CSV/Excel":
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        st.session_state["df"] = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

# --- Dataverse Connection ---
elif data_source == "Dataverse":
    st.sidebar.subheader("Dataverse Connection")
    
    # User input fields for Dataverse credentials
    tenant_id = st.sidebar.text_input(
        "Tenant ID",
        help="Your Azure AD Tenant ID (GUID format)",
        
    )
    
    client_id = st.sidebar.text_input(
        "Client ID (Application ID)",
        help="Your App Registration Client ID",
        
    )
    
    client_secret = st.sidebar.text_input(
        "Client Secret",
        type="password",
        help="Your App Registration Client Secret",
        placeholder="Enter your client secret"
    )
    
    resource_url = st.sidebar.text_input(
        "Dataverse Environment URL",
        help="Your Dataverse environment URL",
        placeholder="e.g. https://orgname.crm4.dynamics.com/"
    )
    
    # Only proceed if all fields are filled
    if tenant_id and client_id and client_secret and resource_url:
        try:
            with st.spinner("Connecting to Dataverse..."):
                access_token = get_access_token(tenant_id, client_id, client_secret, resource_url)
                tables = get_tables(access_token, resource_url)
                table_options = {f"{t['EntitySetName']} ({t['DisplayName']})": t["EntitySetName"] for t in tables}
            st.sidebar.success(f"✓ Connected! Found {len(tables)} custom tables")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")
            tables = []
            table_options = {}
    else:
        st.sidebar.info("Please enter all Dataverse connection details above")
        tables = []
        table_options = {}

    if table_options:
        table_display = st.sidebar.selectbox("Select Table", options=[""] + list(table_options.keys()), index=0)
        table_name = table_options.get(table_display)

        # Get the logical name for field fetching (try singular if plural)
        field_fetch_name = table_name
        if table_name and table_name.endswith("s"):
            field_fetch_name = table_name[:-1]

        if table_name:
            with st.spinner("Fetching table fields..."):
                fields = get_fields(field_fetch_name, access_token, resource_url)
                
            if fields:
                st.sidebar.success(f"Found {len(fields)} fields")
                
                # Show field breakdown
                primary_fields = [f for f in fields if f.get("IsPrimaryId", False)]
                custom_fields = [f for f in fields if f.get("IsCustomAttribute", False)]
                system_fields = [f for f in fields if not f.get("IsCustomAttribute", False) and not f.get("IsPrimaryId", False)]
                
                with st.sidebar.expander("Field Summary", expanded=False):
                    st.write(f"Primary Key Fields: {len(primary_fields)}")
                    st.write(f"Custom Fields: {len(custom_fields)}")
                    st.write(f"System Fields: {len(system_fields)}")
                
                # Test field access to avoid 400 errors
                with st.spinner("Testing field accessibility..."):
                    accessible_fields = test_field_access(table_name, fields, access_token, resource_url)
                
                if accessible_fields:
                    field_options = {f"{f['LogicalName']} ({f['DisplayName']})": f["LogicalName"] for f in accessible_fields}
                    
                    # Default selection: primary key + first few custom fields
                    default_selections = []
                    for f in accessible_fields:
                        if f.get("IsPrimaryId", False) or f.get("IsCustomAttribute", False):
                            default_selections.append(f"{f['LogicalName']} ({f['DisplayName']})")
                        if len(default_selections) >= 10:  # Limit to 10 default selections
                            break
                    
                    if not default_selections:
                        default_selections = list(field_options.keys())[:5]
                    
                    selected_fields = st.sidebar.multiselect(
                        "Select Fields",
                        options=list(field_options.keys()),
                        default=default_selections
                    )
                else:
                    st.sidebar.error("No accessible fields found")
                    field_options = {}
                    selected_fields = []
            else:
                st.sidebar.error("No fields found for this table")
                accessible_fields = []
                field_options = {}
                selected_fields = []
        else:
            fields = []
            field_options = {}
            selected_fields = []
            accessible_fields = []

        top = st.sidebar.number_input("Max rows to fetch", min_value=100, max_value=10000, value=1000, step=100)
    else:
        # No tables available or connection failed
        table_name = None
        accessible_fields = []
        field_options = {}
        selected_fields = []

# --- Dataverse Load Table ---
if st.sidebar.button("Load Table") and table_name:
    try:
        # Create connector but bypass the built-in cleaning
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0",
        }

        # Only include fields that exist and were selected
        logical_fields = [field_options[f] for f in selected_fields if f in field_options]
        
        if not logical_fields:
            st.error("No valid fields selected. Please select at least one field.")
            st.stop()
        
        st.info(f"Loading {len(logical_fields)} fields from table '{table_name}'...")
        
        # Make direct API call to avoid DataverseConnector's cleaning
        select_clause = ",".join(logical_fields)
        url = f"{resource_url.rstrip('/')}/api/data/v9.2/{table_name}?$select={select_clause}&$top={top}"
        
        st.info(f"API URL: {url}")
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Debug the raw API response
        json_response = response.json()
        raw_data = json_response.get("value", [])
        
        st.info(f"Raw API response contains {len(raw_data)} records")
        
        # Show raw JSON structure (first record only to avoid clutter)
        with st.expander("Raw API Response Debug", expanded=False):
            st.write(f"Total records in API response: {len(raw_data)}")
            if raw_data:
                st.write("First record structure:")
                st.json(raw_data[0])
            
            # Check for pagination
            if "@odata.nextLink" in json_response:
                st.warning("⚠️ API response is paginated - there are more records available")
                st.write(f"Next link: {json_response['@odata.nextLink']}")
            else:
                st.info("✓ No pagination - all records returned in single response")
        
        # Create DataFrame from raw data
        df_loaded = pd.DataFrame(raw_data)
        
        st.info(f"DataFrame created with {len(df_loaded)} rows and {len(df_loaded.columns)} columns")
        
        # Check for duplicate rows based on primary key if available
        primary_key_field = next((f["LogicalName"] for f in accessible_fields if f.get("IsPrimaryId", False)), None)
        if primary_key_field and primary_key_field in df_loaded.columns:
            duplicates = df_loaded[primary_key_field].duplicated().sum()
            if duplicates > 0:
                st.warning(f"⚠️ Found {duplicates} duplicate primary key values!")
                st.write("Duplicate primary key values:")
                st.write(df_loaded[df_loaded[primary_key_field].duplicated(keep=False)][primary_key_field])
        
        # Apply ONLY display name mapping and basic column cleanup
        field_map = {f["LogicalName"]: f["DisplayName"] for f in accessible_fields}
        df_loaded = apply_display_name_mapping(df_loaded, field_map)
        
        # Only remove @odata columns, don't drop any rows
        odata_cols = [col for col in df_loaded.columns if col.startswith("@odata.")]
        if odata_cols:
            df_loaded = df_loaded.drop(columns=odata_cols, errors='ignore')
        
        # Ensure unique column names without dropping rows
        cols = pd.Series(df_loaded.columns)
        for dup in cols[cols.duplicated()].unique():
            dups = cols[cols == dup].index.tolist()
            for i, idx in enumerate(dups[1:], start=1):
                cols[idx] = f"{dup}_{i}"
        df_loaded.columns = cols
        
        st.session_state["df"] = df_loaded
        st.success(f"Successfully loaded {len(df_loaded)} rows with {len(df_loaded.columns)} columns from {table_display}")
        
    except Exception as e:
        st.error(f"Error loading table: {e}")
        st.error("Try selecting fewer fields or check if you have permission to access this table.")

# ---------------------------
# Display Data Preview
# ---------------------------
df = st.session_state.get("df")
if df is not None:
    st.subheader("Data Preview")
    # Show all rows instead of just head() - this was the issue!
    st.dataframe(df, use_container_width=True)
    st.info(f"Showing all {len(df)} rows and {len(df.columns)} columns")
else:
    st.info("Please upload a CSV or load a Dataverse table to get started.")
    st.stop()

# ---------------------------
# Initialize Chat History
# ---------------------------
if "msgs" not in st.session_state:
    st.session_state["msgs"] = StreamlitChatMessageHistory(key="langchain_messages")
    st.session_state["msgs"].add_ai_message("How can I help you?")
if "plots" not in st.session_state: st.session_state["plots"] = []
if "dataframes" not in st.session_state: st.session_state["dataframes"] = []

msgs = st.session_state["msgs"]

def display_chat_history():
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(st.session_state.plots[plot_index], key=f"history_plot_{plot_index}")
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[df_index], key=f"history_dataframe_{df_index}")
            else:
                st.write(msg.content)

display_chat_history()

# ---------------------------
# Initialize PandasDataAnalyst Agent
# ---------------------------
if "pandas_agent" not in st.session_state:
    st.session_state["pandas_agent"] = PandasDataAnalyst(
        model=llm,
        data_wrangling_agent=DataWranglingAgent(model=llm, log=False, bypass_recommended_steps=True, n_samples=100),
        data_visualization_agent=DataVisualizationAgent(model=llm, n_samples=100, log=False),
    )

pandas_data_analyst = st.session_state["pandas_agent"]

# ---------------------------
# Chat Input and Agent Invocation
# ---------------------------
if question := st.chat_input("Enter your question here:", key="query_input"):
    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        try:
            pandas_data_analyst.invoke_agent(user_instructions=question, data_raw=st.session_state["df"])
            result = pandas_data_analyst.get_response()
        except Exception:
            st.chat_message("ai").write("An error occurred while processing your query. Please try again.")
            msgs.add_ai_message("An error occurred while processing your query. Please try again.")
            st.stop()

        routing = result.get("routing_preprocessor_decision")

        if routing == "chart" and not result.get("plotly_error", False):
            plot_data = result.get("plotly_graph")
            if plot_data:
                plot_json = json.dumps(plot_data) if isinstance(plot_data, dict) else plot_data
                plot_obj = pio.from_json(plot_json)
                plot_index = len(st.session_state.plots)
                st.session_state.plots.append(plot_obj)
                msgs.add_ai_message("Returning the generated chart.")
                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                st.chat_message("ai").write("Returning the generated chart.")
                st.plotly_chart(plot_obj)

        elif routing == "table":
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message("Returning the data table.")
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write("Returning the data table.")
                st.dataframe(data_wrangled)

        else:
            data_wrangled = result.get("data_wrangled")
            if data_wrangled is not None:
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(data_wrangled)
                msgs.add_ai_message("Issue generating chart. Returning data table instead.")
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                st.chat_message("ai").write("Issue generating chart. Returning data table instead.")
                st.dataframe(data_wrangled)
            else:
                msgs.add_ai_message("An error occurred while processing your query.")
                st.chat_message("ai").write("An error occurred while processing your query.")