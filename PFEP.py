import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import math
import re
import io

# --- App Configuration ---
st.set_page_config(page_title="Inventory & Supply Chain Analysis System", layout="wide")

# --- 1. MASTER TEMPLATE AND LOGIC CONSTANTS (Unchanged) ---

ALL_TEMPLATE_COLUMNS = [
    'SR.NO', 'PARTNO', 'PART DESCRIPTION', 'Qty/Veh 1', 'Qty/Veh 2', 'TOTAL', 'UOM', 'ST.NO',
    'FAMILY', 'Qty/Veh 1_Daily', 'Qty/Veh 2_Daily', 'NET', 'UNIT PRICE', 'PART CLASSIFICATION',
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

PFEP_COLUMN_MAP = {
    'part_id': 'PARTNO', 'description': 'PART DESCRIPTION', 'qty_veh': 'Qty/Veh',
    'qty/veh': 'Qty/Veh', 'quantity_per_vehicle': 'Qty/Veh',
    'net_daily_consumption': 'NET', 'unit_price': 'UNIT PRICE',
    'vendor_code': 'VENDOR CODE', 'vendor_name': 'VENDOR NAME', 'city': 'CITY',
    'state': 'STATE', 'country': 'COUNTRY', 'pincode': 'PINCODE',
    'length': 'L-MM_Size', 'width': 'W-MM_Size', 'height': 'H-MM_Size',
    'qty_per_pack': 'QTY/PACK_Sec', 'packing_factor': 'PACKING FACTOR (PF)',
    'primary_packaging_factor': 'PRIMARY PACKING FACTOR'
}

INTERNAL_TO_PFEP_NEW_COLS = {
    'family': 'FAMILY', 'part_classification': 'PART CLASSIFICATION', 'volume_m3': 'Volume (m^3)',
    'size_classification': 'SIZE CLASSIFICATION', 'wh_loc': 'WH LOC'
}

FAMILY_KEYWORD_MAPPING = {
    "ADAPTOR": ["ADAPTOR", "ADAPTER"], "Beading": ["BEADING"],
    "Electrical": ["BATTERY", "HVPDU", "ELECTRICAL", "INVERTER", "SENSOR", "DC", "COMPRESSOR", "TMCS", "COOLING", "BRAKE SIGNAL", "VCU", "VEHICLE CONTROL", "EVCC", "EBS ECU", "ECU", "CONTROL UNIT", "SIGNAL", "TRANSMITTER", "TRACTION", "HV", "KWH", "EBS", "SWITCH", "HORN"],
    "Electronics": ["DISPLAY", "APC", "SCREEN", "MICROPHONE", "CAMERA", "SPEAKER", "DASHBOARD", "ELECTRONICS", "SSD", "WOODWARD", "FDAS", "BDC", "GEN-2", "SENSOR", "BUZZER"],
    "Wheels": ["WHEEL", "TYRE", "TIRE", "RIM"], "Harness": ["HARNESS", "CABLE"], "Mechanical": ["PUMP", "SHAFT", "LINK", "GEAR", "ARM"],
    "Hardware": ["NUT", "BOLT", "SCREW", "WASHER", "RIVET", "M5", "M22", "M12", "CLAMP", "CLIP", "CABLE TIE", "DIN", "ZFP"],
    "Bracket": ["BRACKET", "BRKT", "BKT", "BRCKT"], "ASSY": ["ASSY"], "Sticker": ["STICKER", "LOGO", "EMBLEM"], "Suspension": ["SUSPENSION"],
    "Tank": ["TANK"], "Tape": ["TAPE", "REFLECTOR", "COLOUR"], "Tool Kit": ["TOOL KIT"], "Valve": ["VALVE"], "Hose": ["HOSE"],
    "Insulation": ["INSULATION"], "Interior & Exterior": ["ROLLER", "FIRE", "HAMMER"], "L-angle": ["L-ANGLE"], "Lamp": ["LAMP"], "Lock": ["LOCK"],
    "Lubricants": ["GREASE", "LUBRICANT"], "Medical": ["MEDICAL", "FIRST AID"], "Mirror": ["MIRROR", "ORVM"], "Motor": ["MOTOR"],
    "Mounting": ["MOUNT", "MTG", "MNTG", "MOUNTED"], "Oil": ["OIL"], "Panel": ["PANEL"], "Pillar": ["PILLAR"],
    "Pipe": ["PIPE", "TUBE", "SUCTION", "TUBULAR"], "Plate": ["PLATE"], "Plywood": ["FLOORING", "PLYWOOD", "EPGC"], "Profile": ["PROFILE", "ALUMINIUM"],
    "Rail": ["RAIL"], "Rubber": ["RUBBER", "GROMMET", "MOULDING"], "Seal": ["SEAL"], "Seat": ["SEAT"], "ABS Cover": ["ABS COVER"], "AC": ["AC"],
    "ACP Sheet": ["ACP SHEET"], "Aluminium": ["ALUMINIUM", "ALUMINUM"], "AXLE": ["AXLE"], "Bush": ["BUSH"], "Chassis": ["CHASSIS"],
    "Dome": ["DOME"], "Door": ["DOOR"], "Filter": ["FILTER"], "Flap": ["FLAP"], "FRP": ["FRP", "FACIA"], "Glass": ["GLASS", "WINDSHIELD", "WINDSHILED"],
    "Handle": ["HANDLE", "HAND", "PLASTIC"], "HATCH": ["HATCH"], "HDF Board": ["HDF"]
}
CATEGORY_PRIORITY_FAMILIES = {"ACP Sheet", "ADAPTOR", "Bracket", "Bush", "Flap", "Handle", "Beading", "Lubricants", "Panel", "Pillar", "Rail", "Seal", "Sticker", "Valve"}
BASE_WAREHOUSE_MAPPING = {
    "ABS Cover": "HRR", "ADAPTOR": "MEZ B-01(A)", "Beading": "HRR", "AXLE": "FLOOR", "Bush": "HRR", "Chassis": "FLOOR", "Dome": "MEZ C-02(B)", "Door": "MRR(C-01)",
    "Electrical": "HRR", "Filter": "CRL", "Flap": "MEZ C-02", "Insulation": "MEZ C-02(B)", "Interior & Exterior": "HRR", "L-angle": "MEZ B-01(A)", "Lamp": "CRL",
    "Lock": "CRL", "Lubricants": "HRR", "Medical": "HRR", "Mirror": "HRR", "Motor": "HRR", "Mounting": "HRR", "Oil": "HRR", "Panel": "MEZ C-02", "Pillar": "MEZ C-02",
    "Pipe": "HRR", "Plate": "HRR", "Profile": "HRR", "Rail": "CTR(C-01)", "Seal": "HRR", "Seat": "MRR(C-01)", "Sticker": "MEZ B-01(A)", "Suspension": "MRR(C-01)",
    "Tank": "HRR", "Tool Kit": "HRR", "Valve": "CRL", "Wheels": "HRR", "Hardware": "MEZ B-02(A)", "Glass": "MRR(C-01)", "Harness": "HRR", "Hose": "HRR",
    "Aluminium": "HRR", "ACP Sheet": "MEZ C-02(B)", "Handle": "HRR", "HATCH": "HRR", "HDF Board": "MRR(C-01)", "FRP": "CTR", "Others": "HRR"
}

# --- DISTANCE CALCULATION COMPONENTS (Unchanged) ---
GEOLOCATOR = Nominatim(user_agent="inventory_distance_calculator_streamlit_v3", timeout=10)

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
        except Exception as e:
            st.warning(f"Geocoding exception for '{pincode_str}': {e}")
            if attempt < retries - 1: time.sleep(backoff_factor * (attempt + 1))
            continue
    return (None, None)

def get_distance_code(distance):
    if pd.isna(distance): return None
    elif distance < 50: return 1
    elif distance <= 250: return 2
    elif distance <= 750: return 3
    else: return 4

# --- 2. DATA LOADING AND CONSOLIDATION (Functions Corrected) ---

def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            return pd.read_csv(uploaded_file, low_memory=False)
        elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        st.warning(f"Unsupported file type: {uploaded_file.name}. Please use CSV or Excel.")
        return None
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return None

def find_qty_veh_column(df):
    possible_names = [
        'qty/veh', 'Qty/Veh', 'QTY/VEH', 'qty_veh', 'Qty_Veh', 'quantity/vehicle',
        'Quantity/Vehicle', 'QUANTITY/VEHICLE', 'qty per veh', 'Qty Per Veh',
        'QTY PER VEH', 'vehicle qty', 'Vehicle Qty', 'VEHICLE QTY'
    ]
    for col in df.columns:
        if str(col).lower().strip() in [p.lower() for p in possible_names]:
            return col
    return None

def find_and_rename_columns(df, file_number=None):
    rename_dict, found_keys = {}, []
    qty_veh_col = find_qty_veh_column(df)
    if qty_veh_col:
        if file_number == 1: rename_dict[qty_veh_col] = 'qty_veh_1'
        elif file_number == 2: rename_dict[qty_veh_col] = 'qty_veh_2'
        else: rename_dict[qty_veh_col] = 'qty_veh'
        found_keys.append(rename_dict[qty_veh_col])
    for internal_key, pfep_name in PFEP_COLUMN_MAP.items():
        if internal_key in ['qty_veh', 'qty/veh', 'quantity_per_vehicle']: continue
        for col in df.columns:
            if str(col).lower().strip() == pfep_name.lower():
                rename_dict[col] = internal_key
                found_keys.append(internal_key)
                break
    df.rename(columns=rename_dict, inplace=True)
    if found_keys:
        st.info(f"   Found and mapped columns: {found_keys}")
    else:
        st.warning("   Could not automatically map any standard columns.")
    return df

def process_and_diagnose_qty_columns(df):
    for i in [1, 2]:
        col_name = f'qty_veh_{i}'
        if col_name not in df.columns:
            df[col_name] = 0
        else:
            numeric_col = pd.to_numeric(df[col_name], errors='coerce')
            invalid_count = numeric_col.isna().sum()
            if invalid_count > 0:
                st.warning(f"Found {invalid_count} non-numeric values in 'Qty/Veh {i}'. Setting them to 0.")
            df[col_name] = numeric_col.fillna(0)
    return df

def _consolidate_bom_list(bom_list):
    valid_boms = [df for df in bom_list if 'part_id' in df.columns]
    if not valid_boms:
        return None
    master = valid_boms[0].copy()
    for i, df in enumerate(valid_boms[1:], 2):
        master = pd.merge(master, df, on='part_id', how='outer', suffixes=('', f'_temp{i}'))
        overlap_cols = [c for c in df.columns if f"{c}_temp{i}" in master.columns]
        for col in overlap_cols:
            temp_col = f"{col}_temp{i}"
            master[col] = master[col].fillna(master[temp_col])
            master.drop(columns=[temp_col], inplace=True)
    return master

def _merge_supplementary_df(main_df, new_df):
    if 'part_id' not in new_df.columns: return main_df
    if 'part_id' in main_df.columns:
        main_df = main_df.set_index('part_id')
    else:
        st.error("Error: 'part_id' not found in main DataFrame for merging.")
        return main_df
    new_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
    new_df = new_df.set_index('part_id')
    update_cols = new_df.columns.difference(main_df.columns)
    main_df = main_df.join(new_df[update_cols])
    main_df.update(new_df)
    return main_df.reset_index()

# MODIFIED FUNCTION with better user feedback
def load_and_consolidate_data(uploaded_files, daily_mult_1, daily_mult_2):
    pbom_dfs, mbom_dfs, part_attr_dfs, pkg_dfs = [], [], [], []
    vendor_master_df = None

    with st.spinner("Processing uploaded files..."):
        if 'vendor_master' in uploaded_files and uploaded_files['vendor_master']:
            st.write("Processing Vendor Master file...")
            df = read_uploaded_file(uploaded_files['vendor_master'])
            if df is not None: vendor_master_df = find_and_rename_columns(df)

        if 'packaging' in uploaded_files and uploaded_files['packaging']:
            st.write("Processing Packaging Details file(s)...")
            for f in uploaded_files['packaging']:
                df = read_uploaded_file(f)
                if df is not None: pkg_dfs.append(find_and_rename_columns(df))

        file_type_map = {"PBOM": pbom_dfs, "MBOM": mbom_dfs, "Part Attribute": part_attr_dfs}
        for key, df_list in file_type_map.items():
            internal_key = key.lower().replace(" ", "_")
            if internal_key in uploaded_files and uploaded_files[internal_key]:
                 st.write(f"Processing {key} file(s)...")
                 for i, f in enumerate(uploaded_files[internal_key]):
                     df = read_uploaded_file(f)
                     if df is not None: df_list.append(find_and_rename_columns(df, i + 1 if "BOM" in key else None))

        st.subheader("BOM CONSOLIDATION")
        pbom_master = _consolidate_bom_list(pbom_dfs)
        mbom_master = _consolidate_bom_list(mbom_dfs)

        # --- NEW: Enhanced User Feedback ---
        if pbom_master is not None and mbom_master is not None:
            st.info("Both PBOM and MBOM files were processed. Merging them into a master BOM.")
        elif pbom_master is not None:
            st.info("Only PBOM files were processed. Using them as the master BOM.")
        elif mbom_master is not None:
            st.info("Only MBOM files were processed. Using them as the master BOM.")
        # --- End of New Feedback ---

        master_bom = _consolidate_bom_list([d for d in [pbom_master, mbom_master] if d is not None])

        if master_bom is None or master_bom.empty:
            st.error("CRITICAL ERROR: Could not process BOM files. Please ensure at least one uploaded BOM file (PBOM or MBOM) contains a 'PARTNO' column.")
            return None

        st.success(f"Consolidated BOM base has {master_bom['part_id'].nunique()} unique parts.")

        st.subheader("MERGING DATA")
        final_df = master_bom
        for df in part_attr_dfs + pkg_dfs:
            if df is not None and 'part_id' in df.columns:
                final_df = _merge_supplementary_df(final_df, df)
                st.info("✅ Supplementary part/packaging data merged.")
            else: st.warning("⚠️ Skipping a supplementary file as it lacks 'PARTNO'.")

        if vendor_master_df is not None:
            if 'part_id' in vendor_master_df.columns:
                st.info("Found 'PARTNO' in Vendor Master. Merging data based on Part Number...")
                final_df = _merge_supplementary_df(final_df, vendor_master_df)
                st.success("✅ Vendor data successfully merged.")
            else: st.warning("⚠️ SKIPPED: The uploaded Vendor Master file does not contain a 'PARTNO' column.")
        else: st.info("No Vendor Master file was uploaded. Skipping.")

        final_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
        final_df = process_and_diagnose_qty_columns(final_df)
        final_df['total_qty'] = final_df['qty_veh_1'] + final_df['qty_veh_2']
        final_df['qty_veh_1_daily'] = final_df['qty_veh_1'] * daily_mult_1
        final_df['qty_veh_2_daily'] = final_df['qty_veh_2'] * daily_mult_2
        final_df['net_daily_consumption'] = final_df['qty_veh_1_daily'] + final_df['qty_veh_2_daily']
        st.success("Initial data consolidation and calculation complete!")
    return final_df


# --- (The rest of the file: Classes and main function remain unchanged) ---

# --- 3. PERCENTAGE-BASED PART CLASSIFICATION ---
class PartClassificationSystem:
    def __init__(self):
        self.percentages = {
            'C': {'target': 60, 'tolerance': 5}, 'B': {'target': 25, 'tolerance': 2},
            'A': {'target': 12, 'tolerance': 2}, 'AA': {'target': 3, 'tolerance': 1}
        }
        self.calculated_ranges = {}

    def load_data_from_dataframe(self, df, price_column='unit_price', part_id_column='part_id'):
        self.parts_data = df.copy()
        self.price_column = price_column
        self.part_id_column = part_id_column
        self.calculate_percentage_ranges()

    def calculate_percentage_ranges(self):
        valid_prices = pd.to_numeric(self.parts_data[self.price_column], errors='coerce').dropna().sort_values()
        if valid_prices.empty: return
        total_valid_parts = len(valid_prices)
        st.write(f"Calculating classification ranges from {total_valid_parts} valid prices...")
        ranges, current_idx = {}, 0
        sorted_percentages = sorted(self.percentages.items(), key=lambda item: item[1]['target'])
        for class_name, details in sorted_percentages:
            target_percent = details['target']
            count = round(total_valid_parts * (target_percent / 100))
            end_idx = min(current_idx + count - 1, total_valid_parts - 1)
            if current_idx <= end_idx:
                min_val = valid_prices.iloc[current_idx]
                max_val = valid_prices.iloc[end_idx]
                ranges[class_name] = {'min': min_val, 'max': max_val}
            current_idx = end_idx + 1
        self.calculated_ranges = {k: ranges[k] for k in ['C', 'B', 'A', 'AA'] if k in ranges}
        st.write("   Ranges calculated successfully.")

    def classify_part(self, unit_price):
        try: unit_price = float(unit_price)
        except (ValueError, TypeError): return 'Manual'
        if pd.isna(unit_price): return 'Manual'
        if not self.calculated_ranges: return 'Unclassified'

        if 'AA' in self.calculated_ranges and unit_price >= self.calculated_ranges['AA']['min']: return 'AA'
        if 'A' in self.calculated_ranges and unit_price >= self.calculated_ranges['A']['min']: return 'A'
        if 'B' in self.calculated_ranges and unit_price >= self.calculated_ranges['B']['min']: return 'B'
        if 'C' in self.calculated_ranges and unit_price >= self.calculated_ranges['C']['min']: return 'C'
        if 'C' in self.calculated_ranges and unit_price < self.calculated_ranges['C']['min']: return 'C'

        return 'Unclassified'

    def classify_all_parts(self):
        if self.parts_data is None or not self.calculated_ranges: return None
        return self.parts_data[self.price_column].apply(self.classify_part)

# --- 4. DATA PROCESSING CLASS ---
class ComprehensiveInventoryProcessor:
    def __init__(self, initial_data):
        self.data = initial_data.copy()
        self.rm_days_mapping = {'A1': 4, 'A2': 6, 'A3': 8, 'A4': 11, 'B1': 6, 'B2': 11, 'B3': 13, 'B4': 16, 'C1': 16, 'C2': 31}
        self.classifier = PartClassificationSystem()

    def manual_review_step(self, internal_key, step_name):
        pfep_name = INTERNAL_TO_PFEP_NEW_COLS.get(internal_key, PFEP_COLUMN_MAP.get(internal_key, internal_key))

        if st.checkbox(f"Manually review '{pfep_name}' for {step_name}?", key=f"review_{internal_key}"):
            filename = f"manual_review_{step_name.lower().replace(' ', '_')}.csv"
            review_cols = ['part_id', 'description', internal_key]
            review_df = self.data[[c for c in review_cols if c in self.data.columns]].copy()
            review_df.rename(columns={internal_key: pfep_name, 'part_id': 'PARTNO', 'description': 'PART DESCRIPTION'}, inplace=True)

            st.dataframe(review_df)
            st.download_button(
                label=f"Download '{filename}' for Manual Review",
                data=review_df.to_csv(index=False).encode('utf-8'),
                file_name=filename,
                mime='text/csv',
            )

            uploaded_file = st.file_uploader("Upload the modified review file", type=['csv', 'xlsx'], key=f"upload_{internal_key}")
            if uploaded_file:
                uploaded_df = read_uploaded_file(uploaded_file)
                if uploaded_df is not None and 'PARTNO' in uploaded_df.columns and pfep_name in uploaded_df.columns:
                    uploaded_df.rename(columns={pfep_name: internal_key, 'PARTNO': 'part_id'}, inplace=True)
                    self.data = _merge_supplementary_df(self.data, uploaded_df[['part_id', internal_key]])
                    st.success(f"✅ Manual changes for {step_name} applied.")
                    st.session_state[f'processed_data_{internal_key}'] = self.data
                else:
                    st.error("Upload failed or file is invalid. Please ensure 'PARTNO' and '{pfep_name}' columns exist.")

    def run_family_classification(self):
        st.subheader("(1/6) Family Classification")
        if 'description' not in self.data.columns:
            self.data['family'] = 'Others'; return
        def find_kw_pos(desc, kw):
            match = re.search(r'\b' + re.escape(str(kw).upper()) + r'\b', str(desc).upper())
            return match.start() if match else -1
        def extract_family(desc):
            if pd.isna(desc): return 'Others'
            for fam in CATEGORY_PRIORITY_FAMILIES:
                if fam in FAMILY_KEYWORD_MAPPING and any(find_kw_pos(desc, kw) != -1 for kw in FAMILY_KEYWORD_MAPPING[fam]): return fam
            matches = [(pos, fam) for fam, kws in FAMILY_KEYWORD_MAPPING.items() if fam not in CATEGORY_PRIORITY_FAMILIES for kw in kws for pos in [find_kw_pos(desc, kw)] if pos != -1]
            return min(matches, key=lambda x: x[0])[1] if matches else 'Others'
        self.data['family'] = self.data['description'].apply(extract_family)
        st.success("✅ Automated family classification complete.")
        self.manual_review_step('family', 'Family Classification')

    def run_size_classification(self):
        st.subheader("\n(2/6) Size Classification")
        size_cols = ['length', 'width', 'height']
        if not all(k in self.data.columns for k in size_cols):
            self.data['volume_m3'], self.data['size_classification'] = None, 'Manual'
            st.warning("Missing L/W/H columns. Size classification set to 'Manual'.")
            return
        for col in size_cols: self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.data['volume_m3'] = (self.data['length'] * self.data['width'] * self.data['height']) / 1_000_000_000
        def classify_size(row):
            if pd.isna(row['volume_m3']): return 'Manual'
            dims = [d for d in [row['length'], row['width'], row['height']] if pd.notna(d)]
            if not dims: return 'Manual'
            max_dim = max(dims)
            if row['volume_m3'] > 1.5 or max_dim > 1200: return 'XL'
            if 0.5 < row['volume_m3'] <= 1.5 or 750 < max_dim <= 1200: return 'L'
            if 0.05 < row['volume_m3'] <= 0.5 or 150 < max_dim <= 750: return 'M'
            return 'S'
        self.data['size_classification'] = self.data.apply(classify_size, axis=1)
        st.success("✅ Automated size classification complete.")
        self.manual_review_step('size_classification', 'Size Classification')

    def run_part_classification(self):
        st.subheader("\n(3/6) Part Classification")
        if 'unit_price' not in self.data.columns:
            self.data['part_classification'] = 'Manual'
            st.warning("Missing 'Unit Price' column. Part classification set to 'Manual'.")
            return
        self.classifier.load_data_from_dataframe(self.data)
        self.data['part_classification'] = self.classifier.classify_all_parts()
        st.success("✅ Percentage-based part classification complete.")
        self.manual_review_step('part_classification', 'Part Classification')

    def run_location_based_norms(self, pincode):
        st.subheader(f"(4/6) Distance & Inventory Norms")
        with st.spinner(f"Getting coordinates for location pincode: {pincode}..."):
            current_coords = get_lat_lon(pincode, country="India")
        if current_coords == (None, None):
            st.error(f"CRITICAL: Could not find coordinates for {pincode}. Distances cannot be calculated.")
            self.data['DISTANCE CODE'], self.data['inventory_classification'] = None, None
            self.data['RM IN DAYS'], self.data['RM IN QTY'], self.data['RM IN INR'] = None, None, None
            return
        st.info(f"Processing vendor locations...")
        with st.spinner("Calculating distances to vendors... This may take a while."):
            def calculate_distance(row):
                vendor_coords = get_lat_lon(row.get('pincode'), country="India", city=str(row.get('city', '')).strip(), state=str(row.get('state', '')).strip())
                return geodesic(current_coords, vendor_coords).km if vendor_coords[0] is not None else None
            self.data['distance_km'] = self.data.apply(calculate_distance, axis=1)
        self.data['DISTANCE CODE'] = self.data['distance_km'].apply(get_distance_code)
        def get_inv_class(p, d):
            if pd.isna(p) or pd.isna(d): return None
            d = int(d)
            if p in ['AA', 'A']: return f"A{d}"
            if p == 'B': return f"B{d}"
            if p == 'C': return 'C1' if d in [1, 2] else 'C2'
            return None
        self.data['inventory_classification'] = self.data.apply(lambda r: get_inv_class(r.get('part_classification'), r.get('DISTANCE CODE')), axis=1)
        self.data['RM IN DAYS'] = self.data['inventory_classification'].map(self.rm_days_mapping)
        self.data['RM IN QTY'] = self.data['RM IN DAYS'] * pd.to_numeric(self.data.get('net_daily_consumption'), errors='coerce')
        self.data['RM IN INR'] = self.data['RM IN QTY'] * pd.to_numeric(self.data.get('unit_price'), errors='coerce')
        qty_per_pack = pd.to_numeric(self.data.get('qty_per_pack'), errors='coerce').fillna(1).replace(0, 1)
        packing_factor = pd.to_numeric(self.data.get('packing_factor', 1), errors='coerce').fillna(1)
        self.data['NO OF SEC. PACK REQD.'] = np.ceil(self.data['RM IN QTY'] / qty_per_pack)
        self.data['NO OF SEC REQ. AS PER PF'] = np.ceil(self.data['NO OF SEC. PACK REQD.'] * packing_factor)
        st.success(f"✅ Inventory norms calculated.")
        self.manual_review_step('inventory_classification', 'Inventory Norms')

    def run_warehouse_location_assignment(self):
        st.subheader("\n(5/6) Warehouse Location Assignment")
        if 'family' not in self.data.columns:
            self.data['wh_loc'] = 'HRR'
            st.warning("Missing 'Family' column. Defaulting warehouse location to 'HRR'.")
            return
        def get_wh_loc(row):
            fam, desc, vol_m3 = row.get('family', 'Others'), row.get('description', ''), row.get('volume_m3', None)
            match = lambda w: re.search(r'\b' + re.escape(w) + r'\b', str(desc).upper())
            if fam == "AC" and match("BCS"): return "OUTSIDE"
            if fam in ["ASSY", "Bracket"] and match("STEERING"): return "DIRECT FROM INSTOR"
            if fam == "Electronics" and any(match(k) for k in ["CAMERA", "APC", "MNVR", "WOODWARD"]): return "CRL"
            if fam == "Electrical" and vol_m3 is not None and (vol_m3 * 1_000_000) > 200: return "HRR"
            if fam == "Mechanical" and match("STEERING"): return "DIRECT FROM INSTOR"
            if fam == "Plywood" and not match("EDGE"): return "MRR(C-01)"
            if fam == "Rubber" and match("GROMMET"): return "MEZ B-01"
            if fam == "Tape" and not match("BUTYL"): return "MEZ B-01"
            if fam == "Wheels":
                if match("TYRE") and match("JK"): return "OUTSIDE"
                if match("RIM"): return "MRR(C-01)"
            return BASE_WAREHOUSE_MAPPING.get(fam, "HRR")
        self.data['wh_loc'] = self.data.apply(get_wh_loc, axis=1)
        st.success("✅ Automated warehouse location assignment complete.")
        self.manual_review_step('wh_loc', 'Warehouse Location')

# --- 5. FINAL EXCEL REPORT GENERATION ---
def create_formatted_excel_output(df):
    st.subheader("\n(6/6) Generating Formatted Excel Report")
    with st.spinner("Creating the final Excel report..."):
        final_df = df.copy().loc[:, ~df.columns.duplicated()]
        rename_map = {**PFEP_COLUMN_MAP, **INTERNAL_TO_PFEP_NEW_COLS,
                      'qty_veh_1': 'Qty/Veh 1', 'qty_veh_2': 'Qty/Veh 2', 'total_qty': 'TOTAL',
                      'qty_veh_1_daily': 'Qty/Veh 1_Daily', 'qty_veh_2_daily': 'Qty/Veh 2_Daily',
                      'inventory_classification': 'INVENTORY CLASSIFICATION'}
        final_df.rename(columns={k: v for k, v in rename_map.items() if k in final_df.columns}, inplace=True)
        for col in ALL_TEMPLATE_COLUMNS:
            if col not in final_df.columns: final_df[col] = ''
        final_df = final_df[ALL_TEMPLATE_COLUMNS]
        final_df['SR.NO'] = range(1, len(final_df) + 1)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            h_gray = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'align': 'center', 'fg_color': '#D9D9D9', 'border': 1})
            s_orange = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#FDE9D9', 'border': 1})
            s_blue = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#DCE6F1', 'border': 1})
            final_df.to_excel(writer, sheet_name='Master Data Sheet', startrow=2, header=False, index=False)
            worksheet = writer.sheets['Master Data Sheet']
            headers = [
                ('A1:H1', 'PART DETAILS', h_gray), ('I1:L1', 'Daily consumption', s_orange),
                ('M1:N1', 'PRICE & CLASSIFICATION', s_orange), ('O1:S1', 'Size & Classification', s_orange),
                ('T1:Z1', 'VENDOR DETAILS', s_blue), ('AA1:AO1', 'PACKAGING DETAILS', s_orange),
                ('AP1:AW1', 'INVENTORY NORM', s_blue), ('AX1:BC1', 'WH STORAGE', s_orange),
                ('BD1:BG1', 'SUPPLY SYSTEM', s_blue), ('BH1:BV1', 'LINE SIDE STORAGE', h_gray)
            ]
            for r, t, f in headers: worksheet.merge_range(r, t, f)
            worksheet.write('BW1', ALL_TEMPLATE_COLUMNS[-1], h_gray)
            for col_num, value in enumerate(final_df.columns): worksheet.write(1, col_num, value, h_gray)
            worksheet.set_column('A:A', 6); worksheet.set_column('B:C', 22); worksheet.set_column('D:BW', 18)
        processed_data = output.getvalue()
    st.success(f"✅ Successfully created formatted Excel file!")
    return processed_data

# --- 6. MAIN WORKFLOW ---
def main():
    st.title("🏭 Comprehensive Inventory & Supply Chain Analysis System")

    if 'processing_started' not in st.session_state:
        st.session_state.processing_started = False
    if 'master_df' not in st.session_state:
        st.session_state.master_df = None
    if 'final_report' not in st.session_state:
        st.session_state.final_report = None

    uploaded_files = {}

    st.header("Step 1: Upload Data Files")
    st.info("Select 'Yes' to reveal the uploader for each file type. You must provide at least one BOM file (PBOM or MBOM).")
    
    file_options = [
        ("Vendor Master", "vendor_master", False),
        ("Packaging Details", "packaging", True),
        ("PBOM", "pbom", True),
        ("MBOM", "mbom", True),
        ("Part Attribute", "part_attribute", True)
    ]

    for display_name, key_name, is_multiple in file_options:
        st.markdown(f"**{display_name}**")
        if st.radio(f"Upload a {display_name} file?", ('No', 'Yes'), key=f"radio_{key_name}", horizontal=True, label_visibility="collapsed") == 'Yes':
            uploaded_files[key_name] = st.file_uploader(
                f"Upload {display_name} File(s)",
                type=['csv', 'xlsx'],
                accept_multiple_files=is_multiple,
                key=f"upload_{key_name}"
            )

    st.markdown("---")
    
    st.header("Step 2: Enter Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        daily_mult_1 = st.number_input("Daily production for Vehicle Type 1", min_value=0.0, value=1.0, step=0.1)
    with col2:
        daily_mult_2 = st.number_input("Daily production for Vehicle Type 2", min_value=0.0, value=1.0, step=0.1)
    with col3:
        pincode = st.text_input("Your location's pincode", value="411001")
    
    st.markdown("---")

    st.header("Step 3: Run Analysis")
    if st.button("🚀 Start Comprehensive Analysis"):
        has_pbom = 'pbom' in uploaded_files and uploaded_files['pbom']
        has_mbom = 'mbom' in uploaded_files and uploaded_files['mbom']

        if not has_pbom and not has_mbom:
            st.error("You must upload at least one PBOM or MBOM file to start the analysis.")
        else:
            st.session_state.processing_started = True
            st.session_state.final_report = None
            master_df = load_and_consolidate_data(uploaded_files, daily_mult_1, daily_mult_2)
            if master_df is not None:
                st.session_state.master_df = master_df
            else:
                st.session_state.processing_started = False

    if st.session_state.processing_started and st.session_state.master_df is not None:
        st.markdown("---")
        st.header("Processing Steps & Results")
        processor = ComprehensiveInventoryProcessor(st.session_state.master_df)

        processor.run_family_classification()
        processor.run_size_classification()
        processor.run_part_classification()

        if pincode and pincode.isdigit():
            processor.run_location_based_norms(pincode)
        else:
            st.error("Invalid Pincode provided. Please enter a valid pincode.")

        processor.run_warehouse_location_assignment()

        st.session_state.final_report = create_formatted_excel_output(processor.data)
        st.balloons()
        st.success("🎉 End-to-end process complete!")

    if st.session_state.final_report:
        st.markdown("---")
        st.header("Download Final Report")
        st.download_button(
            label="📥 Download Structured Inventory Data Final.xlsx",
            data=st.session_state.final_report,
            file_name='structured_inventory_data_final.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

if __name__ == "__main__":
    main()
