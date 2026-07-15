import pickle
import numpy as np
from pathlib import Path

# ===================== [User Config Area] Modify parameters here =====================
# Path of original pkl file (required)
INPUT_PKL_PATH = "data/nuscenes/nuscenes_dbinfos_train.pkl"

# Filter threshold for vis_score (entries with value lower than this will be removed)
VIS_SCORE_THRESHOLD = 0.3

# New param: Prefix of OCC data type (auto concatenated as {occ_data_type}_enriched)
OCC_DATA_TYPE = "occ3d"
# ======================================================================

# ===================== [Auto Generate Output Path] No modification needed =====================
def generate_output_path(input_path: str, vis_thr) -> str:
    """
    Automatically generate output file path with filter parameters
    """
    input_path = Path(input_path)
    # Extract basic info of original file
    file_stem = input_path.stem  # File name without extension
    file_suffix = input_path.suffix  # File extension suffix
    parent_dir = input_path.parent  # Parent directory

    # Format vis threshold: replace decimal point with p (0.5 → 0p5)
    vis_str = f"vis_thr_{str(vis_thr).replace('.', 'p')}"

    # Concatenate new file name
    new_filename = f"{file_stem}_{vis_str}{file_suffix}"
    # Generate full output path
    output_path = parent_dir / new_filename
    return str(output_path)


# Auto generate output file path
OUTPUT_PKL_PATH = generate_output_path(INPUT_PKL_PATH, VIS_SCORE_THRESHOLD)


# ======================================================================

def load_pickle_file(file_path: str) -> dict:
    """
    Load pkl file and verify the data type is dictionary
    :param file_path: Path of pkl file
    :return: Loaded dictionary data
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Content of pkl file is not a dictionary! Actual type: {type(data)}")

    print(f"✅ File loaded successfully: {file_path}")
    print(f"✅ Top-level dictionary contains {len(data)} keys")
    return data


def filter_3d_box_data(raw_data: dict) -> dict:
    """
    Core filter logic: filter elements only by vis_score threshold
    :param raw_data: Original dictionary data
    :return: Filtered dictionary (fully consistent with original data structure)
    """
    filtered_data = {}
    # Generate dynamic enriched key name
    enriched_key = f"{OCC_DATA_TYPE}_enriched"

    # Iterate all keys in top-level dictionary
    for top_key, box_list in raw_data.items():
        # Validation: value corresponding to key must be list
        if not isinstance(box_list, list):
            print(f"⚠️  Skip key {top_key}: corresponding value is not a list")
            filtered_data[top_key] = []
            continue

        valid_elements = []
        # Iterate each element in list (each element is a dictionary)
        for idx, elem in enumerate(box_list):
            # Validation: element must be dictionary
            if not isinstance(elem, dict):
                print(f"⚠️  Skip {top_key}[{idx}]: not a dictionary")
                continue

            # Validation: must contain box3d_lidar field (retain original data validation rule)
            if "box3d_lidar" not in elem:
                print(f"⚠️  Skip {top_key}[{idx}]: missing box3d_lidar key")
                continue

            # ===================== Dynamic enriched & vis_score validation =====================
            # 1. Check if dynamically generated enriched key exists
            if enriched_key not in elem:
                print(f"⚠️  Skip {top_key}[{idx}]: missing {enriched_key} key")
                continue
            # 2. Verify enriched content is dictionary
            occ_dict = elem[enriched_key]
            if not isinstance(occ_dict, dict):
                print(f"⚠️  Skip {top_key}[{idx}]: {enriched_key} is not a dictionary")
                continue
            # 3. Check if vis_score key exists
            if "vis_score" not in occ_dict:
                print(f"⚠️  Skip {top_key}[{idx}]: {enriched_key} missing vis_score key")
                continue
            # 4. Verify vis_score is numeric type
            vis_score = occ_dict["vis_score"]
            if not isinstance(vis_score, (int, float)):
                print(f"⚠️  Skip {top_key}[{idx}]: vis_score is not numeric")
                continue
            # ==========================================================================

            # Keep only entries whose vis_score meets threshold requirement
            if vis_score >= VIS_SCORE_THRESHOLD:
                valid_elements.append(elem)

        # Save filtered list
        filtered_data[top_key] = valid_elements
        print(f"📊 {top_key}: Original {len(box_list)} entries → After filter {len(valid_elements)} entries")

    return filtered_data


def save_pickle_file(data: dict, output_path: str):
    """Save filtered dictionary to pkl file"""
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n✅ Filter finished! Result saved to: {output_path}")


def main():
    try:
        # Print configuration info
        print(f"🔧 Current OCC data type: {OCC_DATA_TYPE}, matched key name: {OCC_DATA_TYPE}_enriched")
        print(f"🔧 Auto-generated output path: {OUTPUT_PKL_PATH}")
        # 1. Load original data
        raw_dict = load_pickle_file(INPUT_PKL_PATH)
        # 2. Execute filtering
        filtered_dict = filter_3d_box_data(raw_dict)
        # 3. Save filtered result
        save_pickle_file(filtered_dict, OUTPUT_PKL_PATH)
    except Exception as e:
        print(f"\n❌ Script execution failed: {str(e)}")


if __name__ == "__main__":
    main()
