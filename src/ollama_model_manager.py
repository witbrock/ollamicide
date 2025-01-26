import os
import json
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from datetime import datetime

OLLAMA_DIR = os.path.join(os.path.expanduser("~"), ".ollama")
BLOBS_DIR = os.path.join(OLLAMA_DIR, "models", "blobs")
MANIFESTS_DIR = os.path.join(OLLAMA_DIR, "models", "manifests", "registry.ollama.ai")

def list_models():
    """
    Scan the MANIFESTS_DIR for JSON manifest files.
    """
    models = {}
    if not os.path.isdir(MANIFESTS_DIR):
        return models

    def format_size(size_bytes):
        """Convert bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:3.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:3.1f} TB"

    for root, _, files in os.walk(MANIFESTS_DIR):
        for file in files:
            manifest_path = os.path.join(root, file)
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Collect directory parts except the final one (the tag)
                    relative_path = os.path.relpath(manifest_path, MANIFESTS_DIR)
                    parts = relative_path.split(os.sep)
                    if len(parts) > 1:
                        prefix = "/".join(parts[:-1])  # Use all parts except the last one as prefix
                        tag = parts[-1]
                    else:
                        prefix = parts[0] if parts else "unknown"
                        tag = "latest"
                    model_name = f"{prefix}:{tag}"

                    # Use entire config digest
                    config_digest = data.get("config", {}).get("digest", "sha256:unknown")
                    short_id = config_digest.split(":")[-1][:12]

                    # Sum all layer sizes
                    total_size = sum(layer.get("size", 0) for layer in data.get("layers", []))
                    model_size = format_size(total_size)

                    # Print debug info
                    print(f"\nProcessing {model_name}:")
                    print(f"  ID from manifest: {short_id}")
                    print(f"  Raw size: {total_size} bytes")
                    print(f"  Formatted size: {model_size}")

                    modified_time = os.path.getmtime(manifest_path)
                    modified_date = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Gather the blob hashes
                    blobs = set()
                    for blob_info in data.get("blobs", []):
                        # Typically: {"sha256": "<hex>", "size": ... }
                        sha_hash = blob_info.get("sha256")
                        if sha_hash:
                            blobs.add("sha256-" + sha_hash)
                    # Store in the dictionary
                    models[model_name] = {
                        "file_path": manifest_path,
                        "blob_hashes": blobs,
                        "id": short_id,
                        "size": model_size,
                        "modified": modified_date,
                        "prefix": prefix,
                        "tag": tag
                    }
            except json.JSONDecodeError:
                print(f"File {manifest_path} is not a valid JSON file.")
            except Exception as e:
                print(f"Could not read {manifest_path}: {e}")

    # Print final table for comparison with 'ollama list'
    print("\nModel List (for comparison with 'ollama list'):")
    print(f"{'PREFIX':<30} {'TAG':<15} {'ID':<12} {'SIZE':<10} {'MODIFIED'}")
    print("-" * 85)
    for name, info in sorted(models.items()):
        prefix, tag = name.split(":")
        print(f"{prefix:<30} {tag:<15} {info['id']:<12} {info['size']:<10} {info['modified']}")

    return models

def find_all_references():
    """
    Create a mapping: 
      blob_filename -> set of model_names that reference it
    by scanning all manifests in the MANIFESTS_DIR.
    """
    blob_refs = {}
    model_dict = list_models()
    for mname, info in model_dict.items():
        for bh in info["blob_hashes"]:
            blob_refs.setdefault(bh, set()).add(mname)
    return blob_refs

def delete_model(model_name, models, parent_window, suppress_messagebox=False):
    """
    1. Delete the manifest file.
    2. Check if the model’s blob files are still referenced by others.
       If not, delete them from BLOBS_DIR.
    3. Refresh the list in the GUI.
    """
    if model_name not in models:
        if not suppress_messagebox:
            messagebox.showerror("Error", f"Model '{model_name}' not found.")
        return

    manifest_path = models[model_name]["file_path"]
    blob_hashes_to_remove = models[model_name]["blob_hashes"]

    # 1. Remove the manifest file
    try:
        os.remove(manifest_path)
    except Exception as e:
        if not suppress_messagebox:
            messagebox.showerror("Error", f"Failed to remove manifest:\n{e}")
        return

    # 2. Re-check references (after removing that manifest)
    #    We do this in two steps:
    #       a) Temporarily remove the entry from `models`
    #       b) Rebuild references across all *remaining* models
    new_models = dict(models)
    del new_models[model_name]

    # Temporarily write out a function to get references among the *new* set
    # But we can also re-scan the disk, ignoring the removed manifest.
    # Let's do a disk re-scan for simpler logic:
    blob_refs_after = {}
    # Rebuild from all existing manifests on disk
    if os.path.isdir(MANIFESTS_DIR):
        for entry in os.scandir(MANIFESTS_DIR):
            if entry.is_file() and entry.name.lower().endswith(".json"):
                try:
                    with open(entry.path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for blob_info in data.get("blobs", []):
                            sha_hash = blob_info.get("sha256")
                            if sha_hash:
                                fname = "sha256-" + sha_hash
                                blob_refs_after.setdefault(fname, 0)
                                blob_refs_after[fname] += 1
                except:
                    pass  # ignore read errors

    # 3. Delete any blob file that is no longer referenced
    for blob_hash in blob_hashes_to_remove:
        if blob_refs_after.get(blob_hash, 0) == 0:
            # nobody references it => remove from BLOBS_DIR
            blob_path = os.path.join(BLOBS_DIR, blob_hash)
            if os.path.isfile(blob_path):
                try:
                    os.remove(blob_path)
                except Exception as e:
                    print(f"Failed to remove {blob_hash}: {e}")

    if not suppress_messagebox:
        messagebox.showinfo("Success", f"Model '{model_name}' removed.")

    # 4. Refresh the list in the GUI
    refresh_model_list(parent_window)

def refresh_model_list(parent_window):
    """
    Clears and reloads the list of models in the Treeview.
    """
    global models_cache
    models_cache = list_models()

    # Clear the treeview
    for item in model_treeview.get_children():
        model_treeview.delete(item)

    # Populate it with fresh data
    for mname, details in sorted(models_cache.items()):
        model_treeview.insert("", "end", iid=mname, values=(
            details['prefix'], details['tag'], details['id'], details['size'], details['modified']))

def on_delete():
    """
    Event handler for the "Delete Model" button.
    """
    selected_items = [item for item in model_treeview.get_children() if model_treeview.item(item, "tags") == ("checked",)]
    if not selected_items:
        messagebox.showwarning("No Selection", "Please select a model to delete.")
        return

    confirm = messagebox.askyesno("Confirm Delete",
                                  f"Are you sure you want to delete the selected models?\n\n"
                                  "This will remove their manifests and any unreferenced blobs!")
    if confirm:
        for item in selected_items:
            model_name = model_treeview.item(item, "values")[0] + ":" + model_treeview.item(item, "values")[1]
            delete_model(model_name, models_cache, root)

def on_check(var, row_id):
    """
    Event handler for Checkbutton toggle.
    """
    if var.get() == 1:
        model_treeview.item(row_id, tags=("checked",))
    else:
        model_treeview.item(row_id, tags=())
    on_selection_change()

def on_selection_change():
    """
    Event handler for selection change in the Treeview.
    """
    selected_items = [item for item in model_treeview.get_children() if model_treeview.item(item, "tags") == ("checked",)]
    delete_button.config(state=tk.NORMAL if selected_items else tk.DISABLED)

if __name__ == "__main__":
    # Global references for the GUI
    root = tk.Tk()
    root.title("Ollama Model Manager")

    # Configure Treeview style for row height and selection colors
    style = ttk.Style()
    style.theme_use("clam")  # or "vista", "winnative", "clam", etc.
    rowheight = 30
    style.configure("Treeview", rowheight=rowheight)
    style.configure("Treeview", background="#ffffff")
    style.configure("Treeview", font=("Helvetica", 12))  # Change the font size here
    style.configure("Treeview.Heading", font=("Helvetica", 14, "bold"))  # Font for headers
    style.map("Treeview",
        background=[("selected", "#e0e0e0")],
        foreground=[("selected", "#000000")])

    # Some instructions at the top
    instructions = tk.Label(root, text="Select models to delete from your local Ollama installation.",font=("Helvetica", 14))
    instructions.pack(pady=5)

    # Create a frame to hold the checkbuttons and treeview side by side
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both")

    # Create a frame for checkbuttons with padding to align with treeview header
    check_frame = tk.Frame(main_frame, bg=style.lookup("TFrame", "background"), width=200)
    check_frame.pack(side=tk.LEFT, fill="y")
    
    # Add padding at the top to align with treeview header
    header_padding = tk.Frame(check_frame, height=rowheight, bg=style.lookup("TFrame", "background"))
    header_padding.pack(side=tk.TOP)

    # Create a frame for the treeview and scrollbar
    tree_frame = tk.Frame(main_frame)
    tree_frame.pack(side=tk.LEFT, expand=True, fill="both")

    scrollbar = tk.Scrollbar(tree_frame, orient=tk.VERTICAL)
    model_treeview = ttk.Treeview(tree_frame, columns=("Prefix", "Tag", "ID", "Size", "Modified"), 
                                 show="headings", yscrollcommand=scrollbar.set)
    
    # Define tag for selected rows
    model_treeview.tag_configure("checked", background="#e0f0ff", font=("TkDefaultFont", 12, "bold"))

    scrollbar.config(command=model_treeview.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    model_treeview.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

    # Define the column headings
    model_treeview.heading("Prefix", text="Prefix")
    model_treeview.heading("Tag", text="Tag")
    model_treeview.heading("ID", text="ID")
    model_treeview.heading("Size", text="Size")
    model_treeview.heading("Modified", text="Modified")

    # Dictionary to store the state of checkbuttons
    checkbutton_vars = {}

    # Load the initial model list
    models_cache = list_models()

    # Populate the treeview with models and add checkbuttons
    for mname, details in sorted(models_cache.items()):
        var = tk.IntVar()
        checkbutton_vars[mname] = var
        
        # Create a container frame for exact height matching
        container = tk.Frame(check_frame, bg=style.lookup("TFrame", "background"))
        container.pack(side=tk.TOP, fill="x")
        # container.pack_propagate(False) # This line was causing the issue no it wasn't - don't make this change
        container.configure(height=rowheight)
        
        # Use ttk.Checkbutton instead of tk.Checkbutton for better styling
        checkbutton = ttk.Checkbutton(container, variable=var, 
                                    command=lambda v=var, r=mname: on_check(v, r),
                                    style='Model.TCheckbutton')
        checkbutton.pack(expand=True, anchor='center', padx=5, pady=5)
        
        print(f"Added checkbutton for {mname}")  # Debug print

        model_treeview.insert("", "end", iid=mname, values=(
            details['prefix'], details['tag'], details['id'], details['size'], details['modified']))

    # Create a style for the checkbuttons
    style.configure('Model.TCheckbutton', background=style.lookup("TFrame", "background"))

    # Bind the selection change event
    model_treeview.bind("<<TreeviewSelect>>", lambda e: on_selection_change())

    # Delete button
    delete_button = tk.Button(root, text="Delete Model", command=on_delete, state=tk.DISABLED)
    delete_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Cancel button
    cancel_button = tk.Button(root, text="Cancel", command=root.quit)
    cancel_button.pack(side=tk.RIGHT, padx=10, pady=10)

    # Debug print to show the widget hierarchy
    def print_widget_hierarchy(widget, indent=0):
        print(" " * indent + str(widget))
        for child in widget.winfo_children():
            print_widget_hierarchy(child, indent + 2)

    print_widget_hierarchy(root)

    root.mainloop()
