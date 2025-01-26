import os
import sys
import json
from unittest.mock import patch, mock_open
from typing import Any
import pytest
import tkinter as tk  # Import tkinter for GUI elements
from PIL import Image  # Import Pillow for image verification

# Import the module to be tested
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
import ollama_model_manager as omm

# --- Fixtures for creating reusable test resources ---
@pytest.fixture
def test_manifests():
    return {
        "test_model1": {"blobs": [{"sha256": "aaaa"}, {"sha256": "bbbb"}]},
        "test_model2": {"blobs": [{"sha256": "bbbb"}, {"sha256": "cccc"}]},
        "test_model_no_blobs": {"name": "empty_model", "blobs": []},
    }

@pytest.fixture
def manifest_paths(test_manifests: dict[str, Any]):
    paths = {}
    for name, data in test_manifests.items():
        paths[name] = os.path.join(omm.MANIFESTS_DIR, name + ".json")
    return paths

@pytest.fixture
def mock_parent_window(test_manifests: dict[str, Any]):
    if not test_manifests:
        pytest.fail("No models found, skipping UI tests.")
    return None  # Return None instead of creating a Tk window

# --- Tests ---

def test_list_models(test_manifests, manifest_paths, monkeypatch):
    # Mock os.walk
    mock_walk = []
    for name, path in manifest_paths.items():
        dir_path = os.path.dirname(path)
        mock_walk.append((dir_path, [], [os.path.basename(path)]))

    monkeypatch.setattr("ollama_model_manager.os.walk", lambda _: mock_walk)

    # Test with mock files and data
    with patch(
        "ollama_model_manager.open",
        mock_open(read_data=json.dumps(test_manifests["test_model1"])),
    ) as m:
        models = omm.list_models()
        assert m.called

    # Ensure models are found before running UI tests
    if not models:
        pytest.fail("No models found, skipping UI tests.")

    # Assertions for basic functionality (adapt as needed)
    # You might want to check if the 'models' dictionary is populated correctly here

    # Test with empty directory
    monkeypatch.setattr(
        "ollama_model_manager.os.walk", lambda _: []
    )  # Mock an empty directory
    assert (
        omm.list_models() == {}
    )  # Assert that an empty directory returns empty dictionary

def test_find_all_references(monkeypatch):
    mock_model_data = {
        "model1": {"blob_hashes": {"sha256-hash1", "sha256-hash2"}},
        "model2": {"blob_hashes": {"sha256-hash2", "sha256-hash3"}},
    }
    monkeypatch.setattr(omm, "list_models", lambda: mock_model_data)

    expected_blob_refs = {
        "sha256-hash1": {"model1"},
        "sha256-hash2": {"model1", "model2"},
        "sha256-hash3": {"model2"},
    }
    assert omm.find_all_references() == expected_blob_refs

def test_delete_model(test_manifests, manifest_paths, mock_parent_window, monkeypatch):
    # Ensure models are found before running UI tests
    if not test_manifests:
        pytest.fail("No models found, skipping UI tests.")

    model_name = "test_model1"
    models = {
        model_name: {
            "file_path": manifest_paths[model_name],
            "blob_hashes": {"sha256-aaaa", "sha256-bbbb"},
        }
    }

    # Mock messagebox, os.remove, os.walk, and refresh_model_list for testing
    mock_remove = MockRemove()
    monkeypatch.setattr("ollama_model_manager.os.remove", mock_remove)
    mock_refresh = MockRefresh()
    monkeypatch.setattr("ollama_model_manager.refresh_model_list", mock_refresh)

    # Test successful deletion
    monkeypatch.setattr(
        "ollama_model_manager.os.walk", lambda _: []
    )  # Empty dir, no references
    with patch(
        "ollama_model_manager.messagebox.showerror"
    ) as mock_showerror:  # Mock messagebox for error reporting
        omm.delete_model(model_name, models, mock_parent_window, suppress_messagebox=True)
        assert not mock_showerror.called
        assert mock_remove.called_with(manifest_paths[model_name])
        assert mock_refresh.called_with(mock_parent_window)

    # Test when model is not found
    with patch("ollama_model_manager.messagebox.showerror") as mock_showerror:
        omm.delete_model("non_existent_model", models, mock_parent_window, suppress_messagebox=True)
        mock_showerror.assert_called_once_with(
            "Error", "Model 'non_existent_model' not found."
        )

def test_on_delete(monkeypatch):
    # Ensure models are found before running UI tests
    if not hasattr(omm, "models_cache"):
        omm.models_cache = {}

    # Mock the listbox to simulate selection
    mock_listbox = MockListbox()
    monkeypatch.setattr(omm, "model_listbox", mock_listbox)

    # Mock models_cache for delete operation
    mock_models_cache = {"test_model": {}}  # You can add details here if needed
    monkeypatch.setattr(omm, "models_cache", mock_models_cache)

    # Mock the main window (replace root if your application has a differently named main window)
    mock_parent_window = tk.Tk()
    monkeypatch.setattr(omm, "root", mock_parent_window)

    # Test when no selection is made
    mock_listbox.curselection.return_value = ()
    with patch("ollama_model_manager.messagebox.showwarning") as mock_showwarning:
        omm.on_delete()
        mock_showwarning.assert_called_with(
            "No Selection", "Please select a model to delete."
        )

    # Test confirmation dialog (user cancels)
    mock_listbox.curselection.return_value = (0,)
    mock_listbox.get.return_value = (
        "test_model"  # Set the name of the model to be deleted
    )
    with patch(
        "ollama_model_manager.messagebox.askyesno", return_value=False
    ) as mock_askyesno:  # Return False (Cancel)
        with patch(
            "ollama_model_manager.delete_model"
        ) as mock_delete_model:  # Mock delete model function
            omm.on_delete()
            mock_askyesno.assert_called_once()
            assert not mock_delete_model.called  # delete_model should not be called

    # Test successful delete confirmation
    mock_listbox.curselection.return_value = (0,)
    mock_listbox.get.return_value = (
        "test_model"  # Set the name of the model to be deleted
    )
    with patch(
        "ollama_model_manager.messagebox.askyesno", return_value=True
    ) as mock_askyesno:  # Return True (Confirm)
        with patch(
            "ollama_model_manager.delete_model"
        ) as mock_delete_model:  # Mock delete model function
            omm.on_delete()
            mock_delete_model.assert_called_once_with(
                "test_model", mock_models_cache, mock_parent_window
            )

def test_required_dirs_exist():
    assert os.path.isdir(
        omm.MANIFESTS_DIR
    ), f"Manifest directory not found: {omm.MANIFESTS_DIR}"
    assert os.path.isdir(omm.BLOBS_DIR), f"Blob directory not found: {omm.BLOBS_DIR}"

def test_manifests_dir_not_empty():
    assert os.listdir(omm.MANIFESTS_DIR), f"Manifest directory is empty: {omm.MANIFESTS_DIR}"

def test_blobs_dir_not_empty():
    assert os.listdir(omm.BLOBS_DIR), f"Blob directory is empty: {omm.BLOBS_DIR}"

def test_image_files_exist_and_valid():
    """
    Test to verify the existence and validity of the image files.
    """
    script_dir = os.path.dirname(__file__)
    checkbox_unchecked_path = os.path.join(script_dir, "..", "src", "checkbox_unchecked.png")
    checkbox_checked_path = os.path.join(script_dir, "..", "src", "checkbox_checked.png")

    print(f"Checking {checkbox_unchecked_path}")
    print(f"Checking {checkbox_checked_path}")

    # Call load_checkbox_images to ensure the images are created and loaded
    unchecked_img, checked_img = omm.load_checkbox_images()

    assert os.path.exists(checkbox_unchecked_path), f"File not found: {checkbox_unchecked_path}"
    assert os.path.exists(checkbox_checked_path), f"File not found: {checkbox_checked_path}"
    assert os.path.getsize(checkbox_unchecked_path) > 0, f"File is empty: {checkbox_unchecked_path}"
    assert os.path.getsize(checkbox_checked_path) > 0, f"File is empty: {checkbox_checked_path}"

    try:
        with Image.open(checkbox_unchecked_path) as img:
            img.verify()  # Verify that it is, in fact, an image
        with Image.open(checkbox_checked_path) as img:
            img.verify()  # Verify that it is, in fact, an image
    except Exception as e:
        pytest.fail(f"Image verification failed: {e}")

    # Verify that the returned images are PhotoImage instances
    assert isinstance(unchecked_img, tk.PhotoImage), "Unchecked image is not a PhotoImage instance"
    assert isinstance(checked_img, tk.PhotoImage), "Checked image is not a PhotoImage instance"

# --- Mock classes for os.DirEntry, os.remove, and refresh_model_list (replace with more advanced mocks if needed) ---

class MockEntry:
    def __init__(self, path):
        self.path = path

    def is_file(self):
        return True

    @property
    def name(self):
        return os.path.basename(self.path)

class MockRemove:
    def __init__(self):
        self.called = 0
        self.call_args = None

    def __call__(self, *args, **kwargs):
        self.called += 1
        self.call_args = args[0]

    def called_with(self, path):
        return self.call_args == path

class MockRefresh:
    def __init__(self):
        self.called = False
        self.parent = None

    def __call__(self, parent_window):
        self.called = True
        self.parent = parent_window

    def called_with(self, parent_window):
        return self.parent == parent_window

class MockListbox:
    def __init__(self):
        self.items = []

    def curselection(self):
        return (0,)  # Simulate a selection at index 0

    def get(self, index):
        try:
            return self.items[index]
        except IndexError:
            return None

    def insert(self, index, item):
        self.items.insert(index, item)

    def delete(self, first, last=None):
        if last is None:
            del self.items[first]
        else:
            del self.items[first:last]
