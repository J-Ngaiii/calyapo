from calyapo.configurations.data_mappings import FULL_DATA_MAPS
from calyapo.configurations.config import DATA_PATHS


from typing import Dict, Any

def invert_mapping(opt_map: Dict[str, Any]) -> Dict[int, str]:
    """
    Inverts the mapping schema from {Text: Code} to {Code: Text} for O(1) lookup.
    Handles cases where multiple codes map to a single text label (Many-to-One).
    """
    inverted = {}
    for label, code in opt_map.items():
        if isinstance(code, list): # Handle [1, 2] -> '18-29'
            for c in code:
                inverted[c] = label
        else:
            inverted[code] = label
    return inverted

def build_steering_dataset(csv_path: str, output_json_path: str, maps: Dict):
    """
    Parses raw CSV, applies schema mappings, and saves a lightweight JSON.
    """
    df = pd.read_csv(csv_path)
    
    # Pre-compute inverted maps for efficiency
    # Structure: {'partyid': {1: 'Democrat', 4: 'Republican'...}}
    demo_decoders = {
        k: invert_mapping(v) 
        for k, v in maps['IGS']['demo']['opt2text'].items()
    }
    resp_decoders = {
        k: invert_mapping(v) 
        for k, v in maps['IGS']['resp']['opt2text'].items()
    }

    dataset = []

    for _, row in df.iterrows():
        entry = {
            "id": row.get(maps['IGS']['demo']['var2text'].get('ID', 'ID')),
            "steering_context": {},
            "target_response": {}
        }

        # 1. Build Demographic Context (Steering)
        var_map_demo = maps['IGS']['demo']['var2text']
        for csv_col, semantic_name in var_map_demo.items():
            if csv_col == 'ID': continue # Skip ID
            
            raw_val = row.get(csv_col)
            
            # Decode if a decoder exists, else keep raw
            if semantic_name in demo_decoders and raw_val in demo_decoders[semantic_name]:
                decoded_val = demo_decoders[semantic_name][raw_val]
            else:
                decoded_val = raw_val
                
            entry["steering_context"][semantic_name] = decoded_val

        # 2. Build Response Target (Ground Truth)
        var_map_resp = maps['IGS']['resp']['var2text']
        for csv_col, semantic_name in var_map_resp.items():
            raw_val = row.get(csv_col)
            
            if semantic_name in resp_decoders and raw_val in resp_decoders[semantic_name]:
                decoded_val = resp_decoders[semantic_name][raw_val]
            else:
                decoded_val = raw_val
            
            entry["target_response"][semantic_name] = decoded_val

        dataset.append(entry)

    # Serialize to disk
    with open(output_json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Successfully processed {len(dataset)} samples to {output_json_path}")