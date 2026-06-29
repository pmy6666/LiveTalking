import argparse
import glob
import os
import pickle
import sys

import cv2
import numpy as np


VALID_IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def natural_sort_key(value: str):
    parts = []
    current = ""
    is_digit = None
    for ch in value:
        ch_is_digit = ch.isdigit()
        if is_digit is None or ch_is_digit == is_digit:
            current += ch
        else:
            parts.append(int(current) if is_digit else current.lower())
            current = ch
        is_digit = ch_is_digit
    if current:
        parts.append(int(current) if is_digit else current.lower())
    return parts


def collect_images(folder: str):
    files = []
    for pattern in VALID_IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, pattern)))
    files = sorted(files, key=lambda path: natural_sort_key(os.path.basename(path)))
    return files


def load_coords(path: str):
    with open(path, "rb") as f:
        coords = pickle.load(f)
    if not isinstance(coords, list):
        raise ValueError(f"coords.pkl is not a list: {type(coords).__name__}")
    return coords


def validate_image_files(files, label):
    errors = []
    warnings = []

    if not files:
        errors.append(f"{label} is empty")
        return errors, warnings

    seen_names = set()
    for index, path in enumerate(files):
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        if base in seen_names:
            warnings.append(f"{label} has duplicate file name: {base}")
        seen_names.add(base)

        if not stem.isdigit():
            warnings.append(f"{label} file name is not numeric: {base}")

        img = cv2.imread(path)
        if img is None:
            errors.append(f"{label} unreadable image: {path}")
            continue

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            errors.append(f"{label} invalid image size: {path}")

        if index == 0:
            continue

    return errors, warnings


def validate_coords(coords, full_count):
    errors = []
    warnings = []

    if not coords:
        errors.append("coords.pkl is empty")
        return errors, warnings

    if len(coords) != full_count:
        errors.append(f"coords count mismatch: coords={len(coords)} full_imgs={full_count}")

    for idx, item in enumerate(coords[: min(len(coords), full_count)]):
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            errors.append(f"coords[{idx}] is not a 4-element box: {item!r}")
            continue
        y1, y2, x1, x2 = item
        if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in item):
            errors.append(f"coords[{idx}] has non-numeric values: {item!r}")
            continue
        if y2 <= y1 or x2 <= x1:
            errors.append(f"coords[{idx}] has invalid bounds: {item!r}")

    return errors, warnings


def validate_avatar(avatar_dir: str):
    avatar_id = os.path.basename(os.path.normpath(avatar_dir))
    result = {
        "avatar_id": avatar_id,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    full_dir = os.path.join(avatar_dir, "full_imgs")
    face_dir = os.path.join(avatar_dir, "face_imgs")
    coords_path = os.path.join(avatar_dir, "coords.pkl")

    if not os.path.isdir(avatar_dir):
        result["errors"].append(f"avatar dir not found: {avatar_dir}")
        return result

    for required in (full_dir, face_dir):
        if not os.path.isdir(required):
            result["errors"].append(f"missing directory: {required}")

    if not os.path.isfile(coords_path):
        result["errors"].append(f"missing file: {coords_path}")

    if result["errors"]:
        return result

    full_imgs = collect_images(full_dir)
    face_imgs = collect_images(face_dir)
    result["stats"]["full_imgs"] = len(full_imgs)
    result["stats"]["face_imgs"] = len(face_imgs)

    errors, warnings = validate_image_files(full_imgs, "full_imgs")
    result["errors"].extend(errors)
    result["warnings"].extend(warnings)

    errors, warnings = validate_image_files(face_imgs, "face_imgs")
    result["errors"].extend(errors)
    result["warnings"].extend(warnings)

    if len(face_imgs) != len(full_imgs):
        result["errors"].append(f"image count mismatch: full_imgs={len(full_imgs)} face_imgs={len(face_imgs)}")

    try:
        coords = load_coords(coords_path)
        result["stats"]["coords"] = len(coords)
    except Exception as exc:
        result["errors"].append(f"failed to load coords.pkl: {exc}")
        return result

    errors, warnings = validate_coords(coords, len(full_imgs))
    result["errors"].extend(errors)
    result["warnings"].extend(warnings)

    if not result["errors"] and full_imgs and face_imgs:
        sample_full = cv2.imread(full_imgs[0])
        sample_face = cv2.imread(face_imgs[0])
        if sample_full is not None:
            result["stats"]["full_shape"] = list(sample_full.shape[:2])
        if sample_face is not None:
            result["stats"]["face_shape"] = list(sample_face.shape[:2])

    return result


def discover_avatar_dirs(root: str):
    if not os.path.isdir(root):
        return []
    names = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]
    return sorted(names, key=lambda path: natural_sort_key(os.path.basename(path)))


def print_result(result):
    avatar_id = result["avatar_id"]
    print(f"\n[{avatar_id}]")
    if result["stats"]:
        stats = ", ".join(f"{key}={value}" for key, value in result["stats"].items())
        print(f"stats: {stats}")

    if not result["errors"] and not result["warnings"]:
        print("status: OK")
        return

    if result["errors"]:
        print("status: ERROR")
        for item in result["errors"]:
            print(f"  - {item}")
    else:
        print("status: WARN")

    for item in result["warnings"]:
        print(f"  - {item}")


def main():
    parser = argparse.ArgumentParser(description="Validate LiveTalking wav2lip avatar directories")
    parser.add_argument(
        "--avatars-root",
        default=os.path.join("data", "avatars"),
        help="Root directory containing avatar folders",
    )
    parser.add_argument(
        "--avatar-id",
        action="append",
        default=[],
        help="Avatar id to check. Can be used multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all avatar directories under avatars root",
    )
    args = parser.parse_args()

    if args.all:
        avatar_dirs = discover_avatar_dirs(args.avatars_root)
    else:
        target_ids = args.avatar_id or [
            name for name in os.listdir(args.avatars_root)
            if name.startswith("wav2lip") and os.path.isdir(os.path.join(args.avatars_root, name))
        ]
        avatar_dirs = [os.path.join(args.avatars_root, avatar_id) for avatar_id in target_ids]

    if not avatar_dirs:
        print("No avatar directories found to validate.")
        return 1

    error_count = 0
    for avatar_dir in avatar_dirs:
        result = validate_avatar(avatar_dir)
        print_result(result)
        if result["errors"]:
            error_count += 1

    print("\nSummary:")
    print(f"  checked: {len(avatar_dirs)}")
    print(f"  failed:  {error_count}")
    print(f"  passed:  {len(avatar_dirs) - error_count}")
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
