import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import math
import re
import io

# --- 1. MASTER TEMPLATE AND LOGIC CONSTANTS ---

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


# Enhanced column mapping to include qty/veh variations
PFEP_COLUMN_MAP = {
    'part_id': 'PARTNO', 'description': 'PART DESCRIPTION', 'qty_veh': 'Qty/Veh',
    'qty/veh': 'Qty/Veh', 'quantity_per_vehicle': 'Qty/Veh', 'net_daily_consumption': 'NET',
    'unit_price': 'UNIT PRICE', 'vendor_code': 'VENDOR CODE', 'vendor_name': 'VENDOR NAME',
    'city': 'CITY', 'state': 'STATE', 'country': 'COUNTRY', 'pincode': 'PINCODE',
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
    "Mounting": ["MOUNT", "MTG", "MNTG", "MOUNTED"], "Oil": ["OIL"], "Panel": ["PANEL", "PANAL"], "Pillar": ["PILLAR"],
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
WAREHOUSE_LOCATION_FULL_FORMS = {
    "CRL": "CAROUSAL", "HRR": "HIGH RACK", "MEZ B-01": "MEZANNAINE B-01", "MEZ B-01(A)": "MEZANNAINE B-01(A)", "MEZ B-02": "MEZANNAINE B-02",
    "MEZ B-02(A)": "MEZANNAINE B-02(A)", "MEZ C-02": "MEZANNAINE C-02", "MEZ C-02(B)": "MEZANNAINE C-02(B)", "MRR": "MID RISE RACK",
    "MRR(C-01)": "MID RISE RACK (C-01)", "CTR": "CANTILEVER RACK (FASCIA)", "CTR(C-01)": "CANTILEVER RACK (C-01)", "FLOOR": "FLOOR", "OUTSIDE": "OUTSIDE",
    "DIRECT FROM INSTOR": "DIRECT FROM INSTOR"
}


# --- DISTANCE CALCULATION COMPONENTS ---
@st.cache_data
def get_lat_lon(_geolocator, pincode, country="India", city="", state="", retries=3, backoff_factor=2):
    pincode_str = str(pincode).strip().split('.')[0]
    if not pincode_str.isdigit() or int(pincode_str) == 0:
        return (None, None)

    query = f"{pincode_str}, {country}"
    if city and state:
        query = f"{pincode_str}, {city}, {state}, {country}"

    for attempt in range(retries):
        try:
            time.sleep(1)
            location = _geolocator.geocode(query, timeout=10)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff_factor * (attempt + 1))
            continue
    return (None, None)

def get_distance_code(distance):
    if pd.isna(distance): return None
    if distance < 50: return 1
    if distance <= 250: return 2
    if distance <= 750: return 3
    return 4

# --- 2. DATA LOADING AND CONSOLIDATION ---
def read_uploaded_file(uploaded_file):
    """Reads an uploaded file into a pandas DataFrame."""
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
        if col in possible_names or str(col).lower().strip() in [p.lower() for p in possible_names]:
            return col
    return None

def find_and_rename_columns(df, file_number=None):
    rename_dict, found_keys = {}, []
    qty_veh_col = find_qty_veh_column(df)
    if qty_veh_col:
        target_col = {1: 'qty_veh_1', 2: 'qty_veh_2'}.get(file_number, 'qty_veh')
        rename_dict[qty_veh_col] = target_col
        found_keys.append(target_col)

    for internal_key, pfep_name in PFEP_COLUMN_MAP.items():
        if internal_key in ['qty_veh', 'qty/veh', 'quantity_per_vehicle']: continue
        if pfep_name in df.columns:
            rename_dict[pfep_name] = internal_key
            found_keys.append(internal_key)
    df.rename(columns=rename_dict, inplace=True)
    return df, 'part_id' in found_keys

def process_and_diagnose_qty_columns(df):
    log = []
    for col_name, default_name in [('qty_veh_1', 'Qty/Veh 1'), ('qty_veh_2', 'Qty/Veh 2')]:
        if col_name not in df.columns:
            log.append(f"INFO: '{default_name}' column not found. Creating it with all zeros.")
            df[col_name] = 0
        else:
            numeric_col = pd.to_numeric(df[col_name], errors='coerce')
            invalid_count = numeric_col.isna().sum()
            if invalid_count > 0:
                log.append(f"⚠️ WARNING: Found {invalid_count} rows in '{default_name}' that were blank or non-numeric. These have been set to 0.")
            df[col_name] = numeric_col.fillna(0)
    return df, log

def consolidate_data(bom_files, vendor_file, daily_mult_1, daily_mult_2):
    all_boms, logs = [], []
    file_counter = 0

    for bom_type, files in bom_files.items():
        for file in files:
            df = read_uploaded_file(file)
            if df is None: continue
            file_counter += 1
            df, has_part_id = find_and_rename_columns(df, file_counter if bom_type == "MBOM" else None)
            if has_part_id:
                all_boms.append(df)
                logs.append(f"✅ Added {len(df)} records from '{file.name}'.")
            else:
                logs.append(f"❌ Part ID ('{PFEP_COLUMN_MAP['part_id']}') is essential. Skipping file '{file.name}'.")
                file_counter -= 1

    if not all_boms: return None, logs
    master_bom = all_boms[0].copy()
    for i, df in enumerate(all_boms[1:], 2):
        master_bom = pd.merge(master_bom, df, on='part_id', how='outer', suffixes=('', f'_dup{i}'))
        for col in [c for c in master_bom.columns if f'_dup{i}' in c]:
            original_col = col.replace(f'_dup{i}', '')
            master_bom[original_col] = master_bom[original_col].fillna(master_bom[col])
            master_bom.drop(columns=[col], inplace=True)

    master_bom, qty_logs = process_and_diagnose_qty_columns(master_bom)
    logs.extend(qty_logs)
    master_bom['total_qty'] = master_bom['qty_veh_1'] + master_bom['qty_veh_2']
    master_bom['qty_veh_1_daily'] = master_bom['qty_veh_1'] * daily_mult_1
    master_bom['qty_veh_2_daily'] = master_bom['qty_veh_2'] * daily_mult_2
    master_bom['net_daily_consumption'] = master_bom['qty_veh_1_daily'] + master_bom['qty_veh_2_daily']
    logs.append(f"✅ Qty/Veh calculations complete. Total unique parts: {len(master_bom)}")

    final_df = master_bom
    if vendor_file:
        vendor_df = read_uploaded_file(vendor_file)
        if vendor_df is not None:
            vendor_df, has_part_id = find_and_rename_columns(vendor_df)
            if has_part_id:
                vendor_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
                final_df = pd.merge(master_bom, vendor_df, on='part_id', how='left', suffixes=('', '_vendor'))
                for col in [c for c in final_df.columns if '_vendor' in c]:
                    original_col = col.replace('_vendor', '')
                    final_df[original_col] = final_df[original_col].fillna(final_df[col])
                    final_df.drop(columns=[col], inplace=True)
                logs.append("✅ Vendor data successfully merged.")
            else:
                logs.append("⚠️ Part ID not found in vendor file. Cannot merge.")
    final_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
    return final_df, logs


# --- 3. NEW PERCENTAGE-BASED PART CLASSIFICATION SYSTEM ---
class PartClassificationSystem:
    def __init__(self):
        self.percentages = {'C': {'target': 60, 'tolerance': 5}, 'B': {'target': 25, 'tolerance': 2}, 'A': {'target': 12, 'tolerance': 2}, 'AA': {'target': 3, 'tolerance': 1}}
        self.calculated_ranges = {}
    def load_data_from_dataframe(self, df, price_column='unit_price', part_id_column='part_id'):
        self.parts_data = df.copy()
        self.price_column, self.part_id_column = price_column, part_id_column
        self.calculate_percentage_ranges()
    def calculate_percentage_ranges(self):
        valid_prices = pd.to_numeric(self.parts_data[self.price_column], errors='coerce').dropna().sort_values()
        if valid_prices.empty: return
        total_valid_parts = len(valid_prices)
        ranges, cumulative_percent = {}, 0
        for class_name in ['C', 'B', 'A', 'AA']:
            target_percent = self.percentages[class_name]['target']
            start_idx = int((cumulative_percent / 100) * total_valid_parts)
            end_idx = int(((cumulative_percent + target_percent) / 100) * total_valid_parts) - 1
            start_idx, end_idx = max(0, start_idx), min(end_idx, total_valid_parts - 1)
            if start_idx <= end_idx: ranges[class_name] = {'min': valid_prices.iloc[start_idx], 'max': valid_prices.iloc[end_idx], 'count': end_idx - start_idx + 1}
            else: ranges[class_name] = {'min': None, 'max': None, 'count': 0}
            cumulative_percent += target_percent
        self.calculated_ranges = ranges
    def classify_part(self, unit_price):
        if pd.isna(unit_price): return 'Manual'
        try: unit_price = float(unit_price)
        except (ValueError, TypeError): return 'Manual'
        for class_name in ['AA', 'A', 'B', 'C']:
            range_info = self.calculated_ranges.get(class_name)
            if range_info and range_info['min'] is not None and range_info['min'] <= unit_price <= range_info['max']:
                return class_name
        return 'Unclassified'
    def classify_all_parts(self):
        if not hasattr(self, 'parts_data'): return None
        return self.parts_data[self.price_column].apply(self.classify_part)

# --- 4. DATA PROCESSING CLASS ---
class ComprehensiveInventoryProcessor:
    def __init__(self, initial_data):
        self.data = initial_data
        self.rm_days_mapping = {'A1': 4, 'A2': 6, 'A3': 8, 'A4': 11, 'B1': 6, 'B2': 11, 'B3': 13, 'B4': 16, 'C1': 16, 'C2': 31}
        self.classifier = PartClassificationSystem()
        self.geolocator = Nominatim(user_agent=f"inventory_streamlit_app_{time.time()}")
    def run_family_classification(self):
        if 'description' not in self.data.columns: return
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
    def run_size_classification(self):
        if not all(k in self.data.columns for k in ['length', 'width', 'height']): return
        for key in ['length', 'width', 'height']: self.data[key] = pd.to_numeric(self.data[key], errors='coerce')
        self.data['volume_m3'] = (self.data['length']/1000 * self.data['width']/1000 * self.data['height']/1000)
        def classify_size(row):
            if pd.isna(row['volume_m3']): return 'Manual'
            dims = [d for d in [row['length'], row['width'], row['height']] if pd.notna(d)]
            if not dims: return 'Manual'
            max_dim = max(dims)
            if row['volume_m3'] > 1.5 or max_dim > 1200: return 'XL'
            if (0.5 < row['volume_m3'] <= 1.5) or (750 < max_dim <= 1200): return 'L'
            if (0.05 < row['volume_m3'] <= 0.5) or (150 < max_dim <= 750): return 'M'
            return 'S'
        self.data['size_classification'] = self.data.apply(classify_size, axis=1)
    def run_part_classification(self):
        if 'unit_price' not in self.data.columns or 'part_id' not in self.data.columns:
            self.data['part_classification'] = 'Manual'
            return
        self.classifier.load_data_from_dataframe(self.data)
        self.data['part_classification'] = self.classifier.classify_all_parts()
    def calculate_distances_for_location(self, current_pincode):
        current_coords = get_lat_lon(self.geolocator, current_pincode, country="India")
        if current_coords == (None, None):
            st.error(f"CRITICAL: Could not find coordinates for your pincode {current_pincode}. Distances cannot be calculated.")
            return [None] * len(self.data)
        st.success(f"Current location found at: {current_coords}")

        progress_bar = st.progress(0)
        status_text = st.empty()

        distances, distance_codes = [], []
        for col in ['pincode', 'city', 'state']:
            if col not in self.data.columns: self.data[col] = ''

        total_vendors = len(self.data)
        for i, (idx, row) in enumerate(self.data.iterrows()):
            pincode_str = str(row.get('pincode', '')).strip().split('.')[0]
            if not pincode_str or not pincode_str.isdigit() or int(pincode_str) == 0:
                distances.append(None); distance_codes.append(None)
                continue
            vendor_coords = get_lat_lon(self.geolocator, pincode_str, country="India", city=str(row.get('city', '')).strip(), state=str(row.get('state', '')).strip())
            if vendor_coords == (None, None):
                distances.append(None); distance_codes.append(None)
            else:
                try:
                    distance_km = geodesic(current_coords, vendor_coords).km
                    distances.append(distance_km)
                    distance_codes.append(get_distance_code(distance_km))
                except Exception:
                    distances.append(None); distance_codes.append(None)

            progress = (i + 1) / total_vendors
            progress_bar.progress(progress)
            status_text.text(f"Processing vendor locations: {i + 1}/{total_vendors}")

        status_text.text("Distance calculation complete.")
        return distance_codes
    def run_location_based_norms(self, location_name, pincode):
        self.data['DISTANCE CODE'] = self.calculate_distances_for_location(pincode)
        if 'part_classification' not in self.data.columns: return
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
        self.data['PACKING FACTOR (PF)'] = pd.to_numeric(self.data.get('packing_factor', 1), errors='coerce').fillna(1)
        qty_per_pack = pd.to_numeric(self.data.get('qty_per_pack'), errors='coerce').fillna(1).replace(0, 1)
        self.data['NO OF SEC. PACK REQD.'] = np.ceil(self.data['RM IN QTY'] / qty_per_pack)
        self.data['NO OF SEC REQ. AS PER PF'] = np.ceil(self.data['NO OF SEC. PACK REQD.'] * self.data['PACKING FACTOR (PF)'])
    def run_warehouse_location_assignment(self):
        if 'family' not in self.data.columns: return
        def get_wh_loc(row):
            fam, desc, vol_m3 = row.get('family', 'Others'), row.get('description', ''), row.get('volume_m3', None)
            word_match = lambda text, word: re.search(r'\b' + re.escape(word) + r'\b', str(text).upper())
            if fam == "AC": return "OUTSIDE" if word_match(desc, "BCS") else "MEZ C-02(B)"
            if fam in ["ASSY", "Bracket"] and word_match(desc, "STEERING"): return "DIRECT FROM INSTOR"
            if fam == "Bracket": return "HRR"
            if fam == "Electronics": return "CRL" if any(word_match(desc, k) for k in ["CAMERA", "APC", "MNVR", "WOODWARD"]) else "HRR"
            if fam == "Electrical": return "CRL" if vol_m3 is not None and (vol_m3 * 1_000_000) > 200 else "HRR"
            if fam == "Mechanical": return "DIRECT FROM INSTOR" if word_match(desc, "STEERING") else "HRR"
            if fam == "Plywood": return "HRR" if word_match(desc, "EDGE") else "MRR(C-01)"
            if fam == "Rubber": return "MEZ B-01" if word_match(desc, "GROMMET") else "HRR"
            if fam == "Tape": return "HRR" if word_match(desc, "BUTYL") else "MEZ B-01"
            if fam == "Wheels":
                if word_match(desc, "TYRE") and word_match(desc, "JK"): return "OUTSIDE"
                if word_match(desc, "RIM"): return "MRR(C-01)"
            return BASE_WAREHOUSE_MAPPING.get(fam, "HRR")
        self.data['wh_loc'] = self.data.apply(get_wh_loc, axis=1)


# --- 5. FINAL EXCEL REPORT GENERATION ---
@st.cache_data
def create_formatted_excel_output(_df):
    output = io.BytesIO()
    final_df = _df.copy().loc[:, ~_df.columns.duplicated()]
    enhanced_map = {'qty_veh_1': 'Qty/Veh 1', 'qty_veh_2': 'Qty/Veh 2', 'total_qty': 'TOTAL', 'qty_veh_1_daily': 'Qty/Veh 1_Daily', 'qty_veh_2_daily': 'Qty/Veh 2_Daily', **PFEP_COLUMN_MAP, **INTERNAL_TO_PFEP_NEW_COLS, 'inventory_classification': 'INVENTORY CLASSIFICATION'}
    final_df.rename(columns={k: v for k, v in enhanced_map.items() if k in final_df.columns}, inplace=True)
    for col in [c for c in ALL_TEMPLATE_COLUMNS if c not in final_df.columns]: final_df[col] = ''
    final_df = final_df[ALL_TEMPLATE_COLUMNS]
    final_df['SR.NO'] = range(1, len(final_df) + 1)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        h_gray = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'align': 'center', 'fg_color': '#D9D9D9', 'border': 1})
        s_orange = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#FDE9D9', 'border': 1})
        s_blue = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#DCE6F1', 'border': 1})
        
        final_df.to_excel(writer, sheet_name='Master Data Sheet', startrow=2, header=False, index=False)
        worksheet = writer.sheets['Master Data Sheet']
        
        worksheet.merge_range('A1:H1', 'PART DETAILS', h_gray)
        worksheet.merge_range('I1:L1', 'Daily consumption', s_orange)
        worksheet.merge_range('M1:N1', 'PRICE & CLASSIFICATION', s_orange)
        worksheet.merge_range('O1:S1', 'Size & Classification', s_orange)
        worksheet.merge_range('T1:Z1', 'VENDOR DETAILS', s_blue)
        worksheet.merge_range('AA1:AO1', 'PACKAGING DETAILS', s_orange)
        worksheet.merge_range('AP1:AW1', 'INVENTORY NORM', s_blue)
        worksheet.merge_range('AX1:BD1', 'WH STORAGE', s_orange)
        worksheet.merge_range('BE1:BH1', 'SUPPLY SYSTEM', s_blue)
        worksheet.merge_range('BI1:BW1', 'LINE SIDE STORAGE', h_gray)

        for col_num, value in enumerate(final_df.columns): worksheet.write(1, col_num, value, h_gray)
        worksheet.set_column('A:A', 6); worksheet.set_column('B:C', 22); worksheet.set_column('D:BW', 18)
    
    return output.getvalue()

# --- 6. STREAMLIT UI ---
def manual_review_step(df, internal_key, step_name):
    st.subheader(f"Manual Review: {step_name}")
    pfep_name = INTERNAL_TO_PFEP_NEW_COLS.get(internal_key, PFEP_COLUMN_MAP.get(internal_key, internal_key))
    review_df = df[['part_id', 'description', internal_key]].copy()
    review_df.rename(columns={internal_key: pfep_name, 'part_id': 'PARTNO', 'description': 'PART DESCRIPTION'}, inplace=True)
    
    st.dataframe(review_df.head(10)) # Display a preview
    
    csv = review_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Download '{step_name}' file for review",
        data=csv,
        file_name=f"manual_review_{step_name.lower().replace(' ', '_')}.csv",
        mime='text/csv',
    )
    
    uploaded_file = st.file_uploader(f"Upload the modified '{step_name}' file (optional)", type=['csv'], key=f"uploader_{internal_key}")
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if 'PARTNO' in uploaded_df.columns and pfep_name in uploaded_df.columns:
                uploaded_df.rename(columns={pfep_name: internal_key, 'PARTNO': 'part_id'}, inplace=True)
                update_map = uploaded_df.set_index('part_id')[internal_key].to_dict()
                df[internal_key] = df['part_id'].map(update_map).fillna(df[internal_key])
                st.success(f"✅ Manual changes for {step_name} applied.")
                return df
            else:
                st.error("Upload failed. Ensure the file contains 'PARTNO' and the specific classification column.")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            
    return df

def main():
    st.set_page_config(layout="wide", page_title="Inventory & Supply Chain Analysis")

    # --- HEADER ---
    st.title("🏭 PFEP (Plan For Each Part) Analyser")
    st.markdown("Streamline your inventory management by consolidating, classifying, and analyzing parts data to generate a comprehensive PFEP report.")
    st.markdown("---")

    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0

    # --- SIDEBAR FOR NAVIGATION AND STATUS ---
    with st.sidebar:
        st.header("Process Flow")
        steps = [
            "1. Upload & Consolidate",
            "2. Family Classification",
            "3. Size Classification",
            "4. Part Classification",
            "5. Distance & Inventory Norms",
            "6. Warehouse Location",
            "7. Generate Report"
        ]
        
        # Determine the highest completed step to reflect in the sidebar
        active_step_index = -1
        if 'processor' in st.session_state and st.session_state.processor is not None:
            data_cols = st.session_state.processor.data.columns
            if 'family' in data_cols: active_step_index = 1
            if 'size_classification' in data_cols: active_step_index = 2
            if 'part_classification' in data_cols: active_step_index = 3
            if 'inventory_classification' in data_cols: active_step_index = 4
            if 'wh_loc' in data_cols: active_step_index = 5


        for i, step in enumerate(steps):
            # i+1 corresponds to the step number
            if i < active_step_index:
                st.markdown(f"✅ **{step}**")
            elif i == active_step_index:
                st.markdown(f"➡️ **{step}**")
            else:
                st.markdown(f"⏳ _{step}_")

    # --- STEP 1: DATA UPLOAD AND CONSOLIDATION ---
    st.header("Step 1: Upload Data and Set Parameters")
    
    with st.container(border=True):
        st.subheader("📁 Upload Data Files")
        col1, col2, col3 = st.columns(3)
        with col1:
            pbom_files = st.file_uploader("Upload PBOM files", accept_multiple_files=True, type=['csv', 'xlsx'], key='pbom')
        with col2:
            mbom_files = st.file_uploader("Upload MBOM files", accept_multiple_files=True, type=['csv', 'xlsx'], key='mbom')
        with col3:
            vendor_file = st.file_uploader("Upload Vendor Master file", type=['csv', 'xlsx'], key='vendor')

        st.subheader("📊 Set Daily Consumption")
        d_col1, d_col2 = st.columns(2)
        daily_mult_1 = d_col1.number_input("Enter daily production for Vehicle Type 1", min_value=0.0, value=1.0, step=0.1)
        daily_mult_2 = d_col2.number_input("Enter daily production for Vehicle Type 2", min_value=0.0, value=1.0, step=0.1)
        
    if st.button("▶️ **Start Data Consolidation**", type="primary", use_container_width=True):
        if not pbom_files and not mbom_files:
            st.error("No BOM data loaded. Please upload at least one PBOM or MBOM file.")
        else:
            with st.spinner("Consolidating data... Please wait."):
                bom_files = {"PBOM": pbom_files, "MBOM": mbom_files}
                master_df, logs = consolidate_data(bom_files, vendor_file, daily_mult_1, daily_mult_2)
                
                with st.expander("Show Consolidation Log", expanded=False):
                    for log in logs:
                        st.text(log)

                if master_df is not None:
                    st.success("✅ Data consolidation complete! You can now proceed with the processing steps below.")
                    st.session_state.processor = ComprehensiveInventoryProcessor(master_df.loc[:, ~master_df.columns.duplicated()])
                    st.dataframe(master_df.head())
                else:
                    st.error("❌ Data consolidation failed. Please check the logs and your input files.")
                    st.session_state.processor = None # Reset on failure

    # --- SEQUENTIAL PROCESSING STEPS (2 through 7) ---
    if st.session_state.processor:
        processor = st.session_state.processor
        st.markdown("---")
        st.header("Step 2-7: Data Processing and Report Generation")
        
        # Step 2: Family Classification
        with st.container(border=True):
            st.subheader("(2/7) Family Classification")
            if st.button("Run Family Classification", use_container_width=True):
                with st.spinner("Running..."):
                    processor.run_family_classification()
                    st.success("✅ Automated family classification complete.")
            if 'family' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'family', 'Family Classification')

        # Step 3: Size Classification
        with st.container(border=True):
            st.subheader("(3/7) Size Classification")
            if st.button("Run Size Classification", use_container_width=True):
                with st.spinner("Running..."):
                    processor.run_size_classification()
                    st.success("✅ Automated size classification complete.")
            if 'size_classification' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'size_classification', 'Size Classification')

        # Step 4: Part Classification
        with st.container(border=True):
            st.subheader("(4/7) Part Classification (Percentage-Based)")
            if st.button("Run Part Classification", use_container_width=True):
                with st.spinner("Running..."):
                    processor.run_part_classification()
                    st.success("✅ Percentage-based part classification complete.")
                    
                    st.subheader("Calculated Classification Ranges")
                    cols = st.columns(len(processor.classifier.calculated_ranges))
                    for idx, (class_name, info) in enumerate(processor.classifier.calculated_ranges.items()):
                        with cols[idx]:
                            if info['min'] is not None:
                                st.metric(label=f"{class_name} Class ({info['count']} parts)", value=f"₹{info['min']:,.2f} to ₹{info['max']:,.2f}")
                                
            if 'part_classification' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'part_classification', 'Part Classification')

        # Step 5: Inventory Norms
        with st.container(border=True):
            st.subheader("(5/7) Distance & Inventory Norms")
            current_pincode = st.text_input("Enter your current pincode for distance calculation", value="411001", help="Default is Pune")
            if st.button("Run Location-Based Norms", use_container_width=True):
                with st.spinner("Calculating distances and inventory norms... This may take a while."):
                    processor.run_location_based_norms("Pune", current_pincode)
                    st.success("✅ Inventory norms calculated successfully.")
            if 'inventory_classification' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'inventory_classification', 'Inventory Norms')

        # Step 6: Warehouse Location
        with st.container(border=True):
            st.subheader("(6/7) Warehouse Location Assignment")
            if st.button("Run Warehouse Location Assignment", use_container_width=True):
                with st.spinner("Running..."):
                    processor.run_warehouse_location_assignment()
                    st.success("✅ Automated warehouse location assignment complete.")
            if 'wh_loc' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'wh_loc', 'Warehouse Location')

        # Step 7: Final Report
        with st.container(border=True):
            st.subheader("(7/7) Generate Final Report")
            st.markdown("Once all previous steps are completed and reviewed, click the button below to generate the final formatted Excel report.")
            if st.button("✅ **Generate Formatted Excel Report**", type="primary", use_container_width=True):
                with st.spinner("Generating your Excel report..."):
                    excel_data = create_formatted_excel_output(processor.data)
                    st.success("Report generated successfully!")
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_data,
                        file_name="structured_inventory_data_final.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    st.balloons()
                    st.markdown("---")
                    st.markdown("🎉 **End-to-end process complete!**")

if __name__ == "__main__":
    main()
