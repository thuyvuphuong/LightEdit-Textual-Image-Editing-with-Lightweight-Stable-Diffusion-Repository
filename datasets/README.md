### 📂 Download Datasets

You can use the following datasets for training or evaluation:

- **Full InstructPix2Pix Dataset**  
  📦 [`timbrooks/instructpix2pix-clip-filtered`](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered)

- **Small Sample Set (1,000 samples)**  
  📦 [`fusing/instructpix2pix-1000-samples`](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples)

- **MagicBrush Dataset (Custom)**  
  📦 [`osunlp/MagicBrush`](https://huggingface.co/datasets/osunlp/MagicBrush)

> ⚠️ When using the **MagicBrush** dataset, make sure to adjust the column names in your script. Here's a snippet to help you handle dataset-specific column mapping:

```python
# 6. Get the column names for input/target.
dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)

if args.original_image_column is None:
    original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
else:
    original_image_column = args.original_image_column
    if original_image_column not in column_names:
        raise ValueError(
            f"'--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
        )

if args.edit_prompt_column is None:
    edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
else:
    edit_prompt_column = args.edit_prompt_column
    if edit_prompt_column not in column_names:
        raise ValueError(
            f"'--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
        )

if args.edited_image_column is None:
    edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
else:
    edited_image_column = args.edited_image_column
    if edited_image_column not in column_names:
        raise ValueError(
            f"'--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
        )
