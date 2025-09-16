import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import math
import re
import io
from functools import reduce

# --- App Configuration ---
st.set_page_config(page_title="Inventory & Supply Chain Analysis System", layout="wide")

# --- 1. MASTER TEMPLATE AND LOGIC CONSTANTS ---
BASE_TEMPLATE_COLUMNS = [
    'SR.NO', 'PARTNO', 'PART DESCRIPTION', 'UOM', 'ST.NO', 'FAMILY', 'NET', 'UNIT PRICE', 'PART CLASSIFICATION',
    'L-MM_Size', 'W-MM_Size', 'H-MM_Size', 'Volume (m^3)', 'SIZE CLASSIFICATION', 'VENDOR CODE',
    'VENDOR NAME', 'VENDOR TYPE', 'CITY', 'STATE', 'COUNTRY', 'PINCODE', 'PRIMARY PACK TYPE',
    'L-MM_Prim_Pack', 'W-MM_Prim_Pack', 'H-MM_Prim_Pack', 'QTY/PACK_Prim', 'PRIM. PACK LIFESPAN',
    'PRIMARY PACKING FACTOR', 'SECONDARY PACK TYPE', 'L-MM_Sec_Pack', 'W-MM_Sec_Pack',
    'H-MM_Sec_Pack', 'NO OF BOXES', 'QTY/PACK_Sec', 'SEC. PACK LIFESPAN', 'ONE WAY/ RETURNABLE',
    'DISTANCE CODE', 'INVENTORY CLASSIFICATION', 'RM IN DAYS', 'RM IN QTY',
    'RM IN INR', 'PACKING FACTOR (PF)', 'NO OF SEC. PACK REQD.', 'NO OF SEC REQ. AS PER PF',
    'WH LOC', 'PRIMARY LOCATION ID', 'SECONDARY LOCATION ID',
    'OVER FLOW TO BE ALLOTED', 'DOCK NUMBER', 'STACKING FACTOR', 'SUPPLY TYPE', 'SUPPLY VEH SET',
    'SUPPLY STRATEGY', 'SUPPLY CONDITION', 'CONTAINER LINE SIDE', 'L-MM_Supply', 'W-MM_Supply',
    'H-MM_Supply', 'Volume_Supply', 'QTY/CONTAINER -LS -9M', 'QTY/CONTAINER -LS-12M', 'STORAGE LINE SIDE',
    'L-MM_Line', 'W-MM_Line', 'H-MM_Line', 'Volume_Line', 'CONTAINER / RACK','NO OF TRIPS/DAY', 'INVENTORY LINE SIDE'
]
PFEP_COLUMN_MAP = { 'part_id': 'PARTNO', 'description': 'PART DESCRIPTION', 'net_daily_consumption': 'NET', 'unit_price': 'UNIT PRICE', 'vendor_code': 'VENDOR CODE', 'vendor_name': 'VENDOR NAME', 'city': 'CITY', 'state': 'STATE', 'country': 'COUNTRY', 'pincode': 'PINCODE', 'length': 'L-MM_Size', 'width': 'W-MM_Size', 'height': 'H-MM_Size', 'qty_per_pack': 'QTY/PACK_Sec', 'packing_factor': 'PACKING FACTOR (PF)', 'primary_packaging_factor': 'PRIMARY PACKING FACTOR' }
INTERNAL_TO_PFEP_NEW_COLS = { 'family': 'FAMILY', 'part_classification': 'PART CLASSIFICATION', 'volume_m3': 'Volume (m^3)', 'size_classification': 'SIZE CLASSIFICATION', 'wh_loc': 'WH LOC', 'inventory_classification': 'INVENTORY CLASSIFICATION' }
FAMILY_KEYWORD_MAPPING = { "ADAPTOR": ["ADAPTOR", "ADAPTER"], "Beading": ["BEADING"], "Electrical": ["BATTERY", "HVPDU", "ELECTRICAL", "INVERTER", "SENSOR", "DC", "COMPRESSOR", "TMCS", "COOLING", "BRAKE SIGNAL", "VCU", "VEHICLE CONTROL", "EVCC", "EBS ECU", "ECU", "CONTROL UNIT", "SIGNAL", "TRANSMITTER", "TRACTION", "HV", "KWH", "EBS", "SWITCH", "HORN"], "Electronics": ["DISPLAY", "APC", "SCREEN", "MICROPHONE", "CAMERA", "SPEAKER", "DASHBOARD", "ELECTRONICS", "SSD", "WOODWARD", "FDAS", "BDC", "GEN-2", "SENSOR", "BUZZER"], "Wheels": ["WHEEL", "TYRE", "TIRE", "RIM"], "Harness": ["HARNESS", "CABLE"], "Mechanical": ["PUMP", "SHAFT", "LINK", "GEAR", "ARM"], "Hardware": ["NUT", "BOLT", "SCREW", "WASHER", "RIVET", "M5", "M22", "M12", "CLAMP", "CLIP", "CABLE TIE", "DIN", "ZFP"], "Bracket": ["BRACKET", "BRKT", "BKT", "BRCKT"], "ASSY": ["ASSY"], "Sticker": ["STICKER", "LOGO", "EMBLEM"], "Suspension": ["SUSPENSION"], "Tank": ["TANK"], "Tape": ["TAPE", "REFLECTOR", "COLOUR"], "Tool Kit": ["TOOL KIT"], "Valve": ["VALVE"], "Hose": ["HOSE"], "Insulation": ["INSULATION"], "Interior & Exterior": ["ROLLER", "FIRE", "HAMMER"], "L-angle": ["L-ANGLE"], "Lamp": ["LAMP"], "Lock": ["LOCK"], "Lubricants": ["GREASE", "LUBRICANT"], "Medical": ["MEDICAL", "FIRST AID"], "Mirror": ["MIRROR", "ORVM"], "Motor": ["MOTOR"], "Mounting": ["MOUNT", "MTG", "MNTG", "MOUNTED"], "Oil": ["OIL"], "Panel": ["PANEL"], "Pillar": ["PILLAR"], "Pipe": ["PIPE", "TUBE", "SUCTION", "TUBULAR"], "Plate": ["PLATE"], "Plywood": ["FLOORING", "PLYWOOD", "EPGC"], "Profile": ["PROFILE", "ALUMINIUM"], "Rail": ["RAIL"], "Rubber": ["RUBBER", "GROMMET", "MOULDING"], "Seal": ["SEAL"], "Seat": ["SEAT"], "ABS Cover": ["ABS COVER"], "AC": ["AC"], "ACP Sheet": ["ACP SHEET"], "Aluminium": ["ALUMINIUM", "ALUMINUM"], "AXLE": ["AXLE"], "Bush": ["BUSH"], "Chassis": ["CHASSIS"], "Dome": ["DOME"], "Door": ["DOOR"], "Filter": ["FILTER"], "Flap": ["FLAP"], "FRP": ["FRP", "FACIA"], "Glass": ["GLASS", "WINDSHIELD", "WINDSHILED"], "Handle": ["HANDLE", "HAND", "PLASTIC"], "HATCH": ["HATCH"], "HDF Board": ["HDF"] }
CATEGORY_PRIORITY_FAMILIES = {"ACP Sheet", "ADAPTOR", "Bracket", "Bush", "Flap", "Handle", "Beading", "Lubricants", "Panel", "Pillar", "Rail", "Seal", "Sticker", "Valve"}
BASE_WAREHOUSE_MAPPING = { "ABS Cover": "HRR", "ADAPTOR": "MEZ B-01(A)", "Beading": "HRR", "AXLE": "FLOOR", "Bush": "HRR", "Chassis": "FLOOR", "Dome": "MEZ C-02(B)", "Door": "MRR(C-01)", "Electrical": "HRR", "Filter": "CRL", "Flap": "MEZ C-02", "Insulation": "MEZ C-02(B)", "Interior & Exterior": "HRR", "L-angle": "MEZ B-01(A)", "Lamp": "CRL", "Lock": "CRL", "Lubricants": "HRR", "Medical": "HRR", "Mirror": "HRR", "Motor": "HRR", "Mounting": "HRR", "Oil": "HRR", "Panel": "MEZ C-02", "Pillar": "MEZ C-02", "Pipe": "HRR", "Plate": "HRR", "Profile": "HRR", "Rail": "CTR(C-01)", "Seal": "HRR", "Seat": "MRR(C-01)", "Sticker": "MEZ B-01(A)", "Suspension": "MRR(C-01)", "Tank": "HRR", "Tool Kit": "HRR", "Valve": "CRL", "Wheels": "HRR", "Hardware": "MEZ B-02(A)", "Glass": "MRR(C-01)", "Harness": "HRR", "Hose": "HRR", "Aluminium": "HRR", "ACP Sheet": "MEZ C-02(B)", "Handle": "HRR", "HATCH": "HRR", "HDF Board": "MRR(C-01)", "FRP": "CTR", "Others": "HRR" }

# --- UTILITY FUNCTIONS ---
GEOLOCATOR = Nominatim(user_agent="inventory_distance_calculator_streamlit_v5", timeout=10)
@st.cache_data
def get_lat_lon(pincode, country="India", city="", state="", retries=3, backoff_factor=2):
    pincode_str = str(pincode).strip().split('.')[0]
    if not pincode_str.isdigit() or int(pincode_str) == 0: return (None, None)
    query = f"{pincode_str}, {city}, {state}, {country}" if city and state else f"{pincode_str}, {country}"
    for attempt in range(retries):
        try:
            time.sleep(1)
            location = GEOLOCATOR.geocode(query)
            if location: return (location.latitude, location.longitude)
        except Exception: time.sleep(backoff_factor * (attempt + 1)); continue
    return (None, None)

def get_distance_code(distance):
    if pd.isna(distance): return None
    elif distance < 50: return 1
    elif distance <= 250: return 2
    elif distance <= 750: return 3
    else: return 4

def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith('.csv'): return pd.read_csv(uploaded_file, low_memory=False)
        elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')): return pd.read_excel(uploaded_file)
        return None
    except Exception: return None

# --- DATA LOADING AND CONSOLIDATION ---
def find_and_rename_columns(df):
    rename_dict = {}
    all_known_pfep_names = {**PFEP_COLUMN_MAP, **INTERNAL_TO_PFEP_NEW_COLS}
    for internal_key, pfep_name in all_known_pfep_names.items():
        for col in df.columns:
            if str(col).lower().strip() == pfep_name.lower():
                rename_dict[col] = internal_key; break
    qty_veh_regex = re.compile(r'(qty|quantity)[\s_/]?p?e?r?[\s_/]?veh(icle)?', re.IGNORECASE)
    for original_col in [c for c in df.columns if qty_veh_regex.search(str(c))]:
        if original_col not in rename_dict:
            rename_dict[original_col] = f"qty_veh_temp_{original_col}"
    df.rename(columns=rename_dict, inplace=True)
    return df

def _consolidate_dataframes(list_of_dfs):
    valid_dfs = [df for df in list_of_dfs if 'part_id' in df.columns]
    if not valid_dfs: return None
    
    def merge_logic(left_df, right_df):
        merged = pd.merge(left_df, right_df, on='part_id', how='outer', suffixes=('_x', None))
        for col in [c for c in right_df.columns if f"{c}_x" in merged.columns]:
            merged[col] = merged[col].fillna(merged[f"{col}_x"])
            merged.drop(columns=[f"{col}_x"], inplace=True)
        return merged
        
    master_df = reduce(merge_logic, valid_dfs)
    return master_df

def _merge_supplementary_df(main_df, new_df):
    if 'part_id' not in new_df.columns:
        st.warning("Uploaded file for merging was skipped because it doesn't contain a 'PARTNO' column.")
        return main_df
    
    main_df_indexed = main_df.set_index('part_id')
    new_df_indexed = new_df.drop_duplicates(subset=['part_id'], keep='first').set_index('part_id')
    main_df_indexed.update(new_df_indexed)
    new_parts = new_df_indexed.index.difference(main_df_indexed.index)
    if not new_parts.empty:
        main_df_indexed = pd.concat([main_df_indexed, new_df_indexed.loc[new_parts]])
    return main_df_indexed.reset_index()

def initial_data_load_and_detect(uploaded_files):
    all_dfs = []
    with st.spinner("Processing uploaded files and detecting vehicle columns..."):
        for files in uploaded_files.values():
            if files:
                for f in (files if isinstance(files, list) else [files]):
                    df = read_uploaded_file(f)
                    if df is not None: all_dfs.append(find_and_rename_columns(df))
        
        if not any('part_id' in df.columns for df in all_dfs):
            st.error("CRITICAL ERROR: No 'PARTNO' column found in any of the uploaded files. Cannot proceed.")
            return None, None
        
        master_df = _consolidate_dataframes(all_dfs)
        if master_df is None or master_df.empty:
            st.error("CRITICAL ERROR: Failed to consolidate files."); return None, None
        
        master_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
        detected_qty_cols = sorted([col for col in master_df.columns if 'qty_veh_temp_' in col])
        rename_map = {old_name: f"qty_veh_{i}" for i, old_name in enumerate(detected_qty_cols)}
        master_df.rename(columns=rename_map, inplace=True)
        final_qty_cols = sorted(rename_map.values())

        for col in final_qty_cols:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)

        st.success(f"Consolidated data and detected {len(final_qty_cols)} 'Quantity per Vehicle' columns.")
        return master_df, final_qty_cols

# --- DATA PROCESSING CLASSES ---
class PartClassificationSystem:
    def __init__(self):
        self.percentages = {'AA': 3, 'A': 12, 'B': 25, 'C': 60}
        self.calculated_ranges = {}

    def calculate_percentage_ranges(self, df, price_column='unit_price'):
        valid_prices = pd.to_numeric(df[price_column], errors='coerce').dropna().sort_values(ascending=False)
        if valid_prices.empty: return
        
        total_parts, current_idx = len(valid_prices), 0
        for class_name, percentage in self.percentages.items():
            count = round(total_parts * (percentage / 100))
            end_idx = min(current_idx + count, total_parts)
            if class_name == 'C': end_idx = total_parts
            if current_idx < end_idx:
                self.calculated_ranges[class_name] = {'min': valid_prices.iloc[end_idx - 1]}
            current_idx = end_idx

    def classify_part(self, unit_price):
        if pd.isna(unit_price): return 'Manual'
        if not self.calculated_ranges: return 'Unclassified'
        if 'AA' in self.calculated_ranges and unit_price >= self.calculated_ranges['AA']['min']: return 'AA'
        if 'A' in self.calculated_ranges and unit_price >= self.calculated_ranges['A']['min']: return 'A'
        if 'B' in self.calculated_ranges and unit_price >= self.calculated_ranges['B']['min']: return 'B'
        return 'C'

    def classify_all_parts(self, df, price_column='unit_price'):
        self.calculate_percentage_ranges(df, price_column)
        return df[price_column].apply(self.classify_part)

class ComprehensiveInventoryProcessor:
    def __init__(self, initial_data):
        self.data = initial_data.copy()
        self.rm_days_mapping = {'A1': 4, 'A2': 6, 'A3': 8, 'A4': 11, 'B1': 6, 'B2': 11, 'B3': 13, 'B4': 16, 'C1': 16, 'C2': 31}
        self.classifier = PartClassificationSystem()

    def manual_review_step(self, internal_key, step_name):
        pfep_name = INTERNAL_TO_PFEP_NEW_COLS.get(internal_key, "UNKNOWN_COLUMN")
        st.markdown("---")
        if st.checkbox(f"Manually review '{pfep_name}'?", key=f"review_{internal_key}"):
            review_cols = ['part_id', 'description', internal_key]
            if 'description' not in self.data.columns: review_cols.remove('description')
            review_df = self.data[[c for c in review_cols if c in self.data.columns]].copy()
            rename_map = {'part_id': 'PARTNO', internal_key: pfep_name}
            if 'description' in review_cols: rename_map['description'] = 'PART DESCRIPTION'
            review_df.rename(columns=rename_map, inplace=True)
            
            st.dataframe(review_df)
            st.download_button( f"Download '{step_name}.csv'", review_df.to_csv(index=False).encode('utf-8'), f"review_{step_name}.csv", 'text/csv', key=f"download_{internal_key}" )
            uploaded_file = st.file_uploader("Upload modified review file", type=['csv', 'xlsx'], key=f"upload_{internal_key}")
            if uploaded_file:
                modified_df = read_uploaded_file(uploaded_file)
                if modified_df is not None and 'PARTNO' in modified_df.columns and pfep_name in modified_df.columns:
                    modified_df.rename(columns={'PARTNO': 'part_id', pfep_name: internal_key}, inplace=True)
                    st.session_state.master_df = _merge_supplementary_df(st.session_state.master_df, modified_df[['part_id', internal_key]])
                    st.success(f"✅ Manual changes for {step_name} applied! Refreshing...")
                    time.sleep(2); st.rerun()
                else: st.error(f"Upload failed. Ensure file has 'PARTNO' and '{pfep_name}' columns.")

    def calculate_dynamic_consumption(self, qty_cols, multipliers):
        st.subheader("1. Daily & Net Consumption")
        for i, col in enumerate(qty_cols): self.data[f"{col}_daily"] = self.data[col] * multipliers[i]
        self.data['TOTAL'] = self.data[qty_cols].sum(axis=1)
        self.data['net_daily_consumption'] = self.data[[f"{c}_daily" for c in qty_cols]].sum(axis=1)
        st.success("✅ Consumption calculated.")

    def run_family_classification(self):
        st.subheader("2. Family Classification")
        if 'description' in self.data.columns:
            def extract_family(desc):
                if pd.isna(desc): return 'Others'
                for fam in CATEGORY_PRIORITY_FAMILIES:
                    if fam in FAMILY_KEYWORD_MAPPING and any(re.search(r'\b' + re.escape(kw) + r'\b', str(desc).upper()) for kw in FAMILY_KEYWORD_MAPPING[fam]): return fam
                matches = [(m.start(), fam) for fam, kws in FAMILY_KEYWORD_MAPPING.items() for kw in kws for m in [re.search(r'\b' + re.escape(kw) + r'\b', str(desc).upper())] if m]
                return min(matches, key=lambda x: x[0])[1] if matches else 'Others'
            self.data['family'] = self.data['description'].apply(extract_family)
            st.success("✅ Automated family classification complete.")
        else: self.data['family'] = 'Others'
        self.manual_review_step('family', 'Family_Classification')

    def run_size_classification(self):
        st.subheader("3. Size Classification")
        if all(k in self.data.columns for k in ['length', 'width', 'height']):
            for col in ['length', 'width', 'height']: self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data['volume_m3'] = (self.data['length'] * self.data['width'] * self.data['height']) / 1e9
            def classify_size(row):
                if pd.isna(row['volume_m3']): return 'Manual'
                dims = [d for d in [row['length'], row['width'], row['height']] if pd.notna(d)]
                if not dims: return 'Manual'
                if row['volume_m3'] > 1.5 or max(dims) > 1200: return 'XL'
                if 0.5 < row['volume_m3'] <= 1.5 or 750 < max(dims) <= 1200: return 'L'
                if 0.05 < row['volume_m3'] <= 0.5 or 150 < max(dims) <= 750: return 'M'
                return 'S'
            self.data['size_classification'] = self.data.apply(classify_size, axis=1)
            st.success("✅ Automated size classification complete.")
        else: self.data['volume_m3'], self.data['size_classification'] = None, 'Manual'
        self.manual_review_step('size_classification', 'Size_Classification')

    def run_part_classification(self):
        st.subheader("4. Part Classification")
        if 'unit_price' in self.data.columns:
            self.data['part_classification'] = self.classifier.classify_all_parts(self.data)
            st.success("✅ Percentage-based part classification complete.")
        else: self.data['part_classification'] = 'Manual'
        self.manual_review_step('part_classification', 'Part_Classification')

    def run_location_based_norms(self, pincode):
        st.subheader("5. Distance & Inventory Norms")
        current_coords = get_lat_lon(pincode)
        if current_coords == (None, None): st.error(f"CRITICAL: Could not find coordinates for {pincode}."); return
        self.data['distance_km'] = self.data.apply(lambda r: geodesic(current_coords, get_lat_lon(r.get('pincode'), city=str(r.get('city', '')), state=str(r.get('state', '')))).km if r.get('pincode') and current_coords[0] else None, axis=1)
        self.data['DISTANCE CODE'] = self.data['distance_km'].apply(get_distance_code)
        def get_inv_class(p, d):
            if pd.isna(p) or pd.isna(d): return None
            if p in ['AA', 'A']: return f"A{int(d)}"
            if p == 'B': return f"B{int(d)}"
            return 'C1' if int(d) in [1, 2] else 'C2'
        self.data['inventory_classification'] = self.data.apply(lambda r: get_inv_class(r.get('part_classification'), r.get('DISTANCE CODE')), axis=1)
        self.data['RM IN DAYS'] = self.data['inventory_classification'].map(self.rm_days_mapping)
        self.data['RM IN QTY'] = self.data['RM IN DAYS'] * self.data['net_daily_consumption']
        self.data['RM IN INR'] = self.data['RM IN QTY'] * pd.to_numeric(self.data.get('unit_price'), errors='coerce')
        qty_per_pack = pd.to_numeric(self.data.get('qty_per_pack'), errors='coerce').fillna(1).replace(0, 1)
        packing_factor = pd.to_numeric(self.data.get('packing_factor'), errors='coerce').fillna(1)
        self.data['NO OF SEC. PACK REQD.'] = np.ceil(self.data['RM IN QTY'] / qty_per_pack)
        self.data['NO OF SEC REQ. AS PER PF'] = np.ceil(self.data['NO OF SEC. PACK REQD.'] * packing_factor)
        st.success("✅ Inventory norms calculated.")
        self.manual_review_step('inventory_classification', 'Inventory_Norms')

    def run_warehouse_location_assignment(self):
        st.subheader("6. Warehouse Location Assignment")
        def get_wh_loc(row):
            fam, desc = row.get('family', 'Others'), row.get('description', '')
            if pd.isna(desc): desc = ''
            match = lambda w: re.search(r'\b' + re.escape(w) + r'\b', desc.upper())
            if fam == "AC" and match("BCS"): return "OUTSIDE"
            return BASE_WAREHOUSE_MAPPING.get(fam, "HRR")
        self.data['wh_loc'] = self.data.apply(get_wh_loc, axis=1)
        st.success("✅ Automated warehouse location assignment complete.")
        self.manual_review_step('wh_loc', 'Warehouse_Location')

# --- DYNAMIC EXCEL REPORT GENERATION ---
def create_formatted_excel_output(df, vehicle_configs):
    st.subheader("Generating Formatted Excel Report")
    final_df, num_veh = df.copy(), len(vehicle_configs)
    rename_map = {**PFEP_COLUMN_MAP, **INTERNAL_TO_PFEP_NEW_COLS, 'TOTAL': 'TOTAL'}
    qty_veh_cols, qty_veh_daily_cols = [], []
    for i, config in enumerate(vehicle_configs):
        rename_map[f"qty_veh_{i}"] = config['name']
        rename_map[f"qty_veh_{i}_daily"] = f"{config['name']}_Daily"
        qty_veh_cols.append(config['name'])
        qty_veh_daily_cols.append(f"{config['name']}_Daily")
    final_df.rename(columns=rename_map, inplace=True)
    
    final_cols = BASE_TEMPLATE_COLUMNS[:]
    part_desc_index = final_cols.index('PART DESCRIPTION')
    final_cols[part_desc_index+1:part_desc_index+1] = qty_veh_cols + ['TOTAL']
    family_index = final_cols.index('FAMILY')
    final_cols[family_index+1:family_index+1] = qty_veh_daily_cols
    
    for col in final_cols:
        if col not in final_df.columns: final_df[col] = ''
    final_df = final_df[final_cols]
    final_df['SR.NO'] = range(1, len(final_df) + 1)
    
    with st.spinner("Creating the final Excel report..."):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            h_gray, s_orange, s_blue = (workbook.add_format(f) for f in [{'bold': True, 'text_wrap': True, 'valign': 'top', 'align': 'center', 'fg_color': '#D9D9D9', 'border': 1}, {'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#FDE9D9', 'border': 1}, {'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#DCE6F1', 'border': 1}])
            final_df.to_excel(writer, sheet_name='Master Data Sheet', startrow=2, header=False, index=False)
            worksheet = writer.sheets['Master Data Sheet']
            headers_config = [ ('PART DETAILS', final_cols.index('UOM'), h_gray), ('Daily consumption', final_cols.index('NET') - final_cols.index('FAMILY'), s_orange), ('PRICE & CLASSIFICATION', 2, s_orange), ('Size & Classification', 5, s_orange), ('VENDOR DETAILS', 7, s_blue), ('PACKAGING DETAILS', 15, s_orange), ('INVENTORY NORM', 8, s_blue), ('WH STORAGE', 8, s_orange), ('SUPPLY SYSTEM', 4, s_blue), ('LINE SIDE STORAGE', 15, h_gray) ]
            current_col = 0
            for title, num_cols, style in headers_config:
                if num_cols > 1: worksheet.merge_range(0, current_col, 0, current_col + num_cols - 1, title, style)
                else: worksheet.write(0, current_col, title, style)
                current_col += num_cols
            for col_num, value in enumerate(final_df.columns): worksheet.write(1, col_num, value, h_gray)
            worksheet.set_column('A:A', 6); worksheet.set_column('B:C', 22); worksheet.set_column('D:ZZ', 18)
        return output.getvalue()

# --- MAIN WORKFLOW ---
def main():
    st.title("🏭 Dynamic Inventory & Supply Chain Analysis System")
    if 'app_stage' not in st.session_state: st.session_state.app_stage = "upload"

    if st.session_state.app_stage == "upload":
        st.header("Step 1: Upload Data Files")
        uploaded_files = {}
        file_options = [ ("Vendor Master", "vendor_master", False), ("Packaging Details", "packaging", True), ("PBOM", "pbom", True), ("MBOM", "mbom", True), ("Part Attribute", "part_attribute", True) ]
        for display_name, key, is_multiple in file_options:
            with st.expander(f"Upload {display_name} File(s)"):
                uploaded_files[key] = st.file_uploader(f"Upload", type=['csv', 'xlsx'], accept_multiple_files=is_multiple, key=f"upload_{key}", label_visibility="collapsed")
        st.session_state.pincode = st.text_input("Enter your location's pincode", value="411001")
        if st.button("Detect & Consolidate Files"):
            if not uploaded_files.get("pbom") and not uploaded_files.get("mbom"): st.error("You must upload at least one PBOM or MBOM file.")
            else:
                master_df, qty_cols = initial_data_load_and_detect(uploaded_files)
                if master_df is not None and qty_cols:
                    st.session_state.master_df, st.session_state.qty_cols, st.session_state.app_stage = master_df, qty_cols, "configure"
                    st.rerun()

    if st.session_state.app_stage == "configure":
        st.header("Step 2: Configure Vehicle Types")
        configs = []
        for i, col in enumerate(st.session_state.qty_cols):
            cols = st.columns([2, 1]); name = cols[0].text_input("Vehicle Name", f"Vehicle Type {i+1}", key=f"name_{i}"); multiplier = cols[1].number_input("Daily Production", 1.0, key=f"mult_{i}"); configs.append({"name": name, "multiplier": multiplier})
        if st.button("🚀 Start Full Analysis"):
            st.session_state.vehicle_configs, st.session_state.app_stage = configs, "process_start"
            st.rerun()

    if st.session_state.app_stage and st.session_state.app_stage.startswith("process"):
        st.header("Step 3: Processing & Manual Review")
        processor = ComprehensiveInventoryProcessor(st.session_state.master_df)
        
        stages = ["start", "family", "size", "part", "norms", "wh_loc"]
        stage_str = st.session_state.app_stage.split("_")[-1]
        current_stage_index = stages.index(stage_str) if stage_str in stages else -1
        
        processor.calculate_dynamic_consumption(st.session_state.qty_cols, [c['multiplier'] for c in st.session_state.vehicle_configs])
        if current_stage_index >= 1: processor.run_family_classification()
        if current_stage_index >= 2: processor.run_size_classification()
        if current_stage_index >= 3: processor.run_part_classification()
        if current_stage_index >= 4: processor.run_location_based_norms(st.session_state.pincode)
        if current_stage_index >= 5: processor.run_warehouse_location_assignment()

        st.session_state.master_df = processor.data.copy()

        if current_stage_index < len(stages) - 1:
            next_stage_name = stages[current_stage_index + 1].replace("_", " ").title()
            if st.button(f"Continue to {next_stage_name}"):
                st.session_state.app_stage = f"process_{stages[current_stage_index + 1]}"
                st.rerun()
        else:
            st.markdown("---"); st.info("All processing steps are complete.")
            if st.button("Continue to Final Join Step"):
                st.session_state.app_stage = "join"; st.rerun()
    
    if st.session_state.app_stage == "join":
        st.header("Step 4: Join Additional Data (Optional)")
        st.info("Upload a final file to merge additional data. The file must contain a 'PARTNO' column.")
        join_file = st.file_uploader("Upload Join File", type=['csv', 'xlsx'])
        
        if st.button("Generate Final Report"):
            final_df = st.session_state.master_df.copy()
            if join_file:
                with st.spinner("Merging final data..."):
                    join_df = read_uploaded_file(join_file)
                    if join_df is not None:
                        join_df = find_and_rename_columns(join_df)
                        final_df = _merge_supplementary_df(final_df, join_df)
                        st.success("✅ Final data successfully merged.")
            st.session_state.final_df_joined = final_df
            st.session_state.app_stage = "download"; st.rerun()

    if st.session_state.app_stage == "download":
        st.header("Step 5: Download Final Report")
        report_data = create_formatted_excel_output(st.session_state.final_df_joined, st.session_state.vehicle_configs)
        st.download_button("📥 Download Report", report_data, 'PFEP_Analysis_Report.xlsx')
        if st.button("Start Over"): st.session_state.clear(); st.rerun()

if __name__ == "__main__":
    main()
