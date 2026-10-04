import subprocess
from pathlib import Path
import time


DATASET = "nickj26/places2-mit-dataset"

OUTPUT_DIR = Path("data/external/places365")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_ID = 1
END_ID = 3600

successful = 0
skipped = 0
failed = []

print("=" * 70)
print("DrishtiAI - Places365 Non-Fundus Downloader")
print("=" * 70)
print(f"Target images : {END_ID - START_ID + 1}")
print(f"Output        : {OUTPUT_DIR.resolve()}")
print("=" * 70)

for image_id in range(START_ID, END_ID + 1):

    filename = f"Places365_test_{image_id:08d}.jpg"
    output_file = OUTPUT_DIR / filename

    # Skip already downloaded images
    if output_file.exists() and output_file.stat().st_size > 0:
        skipped += 1
        continue

    kaggle_path = (
        f"test_256/test_256/{filename}"
    )

    command = [
        "kaggle",
        "datasets",
        "download",
        DATASET,
        "-f",
        kaggle_path,
        "-p",
        str(OUTPUT_DIR),
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0 and output_file.exists():

            successful += 1

        else:

            failed.append(
                {
                    "id": image_id,
                    "filename": filename,
                    "error": result.stderr.strip(),
                }
            )

    except Exception as exc:

        failed.append(
            {
                "id": image_id,
                "filename": filename,
                "error": str(exc),
            }
        )

    completed = (
        successful
        + skipped
        + len(failed)
    )

    if completed % 100 == 0:

        print(
            f"Progress: {completed}/{END_ID - START_ID + 1} | "
            f"downloaded={successful} | "
            f"skipped={skipped} | "
            f"failed={len(failed)}"
        )

    # Small delay to avoid hammering the API
    time.sleep(0.05)


print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(f"Downloaded : {successful}")
print(f"Skipped    : {skipped}")
print(f"Failed     : {len(failed)}")

if failed:

    print()
    print("Failed files:")

    for item in failed[:20]:

        print(
            f"{item['id']}: "
            f"{item['filename']}"
        )

    if len(failed) > 20:
        print(
            f"... and {len(failed) - 20} more."
        )

print()
print(
    f"Files currently in directory: "
    f"{len(list(OUTPUT_DIR.glob('*.jpg')))}"
)